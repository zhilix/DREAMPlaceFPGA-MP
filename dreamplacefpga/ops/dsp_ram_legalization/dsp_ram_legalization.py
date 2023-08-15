'''
@File: dsp_ram_legalization.py
@Author: Rachel Selina Rajarathnam (DREAMPlaceFPGA) 
@Date: Oct 2020
'''
import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb 
import numpy as np

import dreamplacefpga.ops.dsp_ram_legalization.legalize_cpp as legalize_cpp
import dreamplacefpga.configure as configure

import logging
logger = logging.getLogger(__name__)

class LegalizeDSPRAMFunction(Function):
    @staticmethod
    def legalize(pos, placedb, region_id, model):
        """
        @brief legalize DSP/RAM at the end of Global Placement
        @param locX Instance locX ndarray
        @param locY Instance locY ndarray
        @param num_nodes Instance count
        @param num_sites Instance site count
        @param sites Instance site ndarray 
        @param precondWL Instance wirelength preconditioner ndarray 
        @param dInit lg_max_dist_init
        @param dIncr lg_max_dist_incr
        @param fScale lg_flow_cost_scale
        @param movVal Maximum & Average Instance movement (list)
        @param outLoc Legalized Instance locations list - {x0, x1, ... xn, y0, y1, ... yn} 
        """
        lg_max_dist_init=10.0
        lg_max_dist_incr=10.0
        lg_flow_cost_scale=100.0
        numNodes = int(pos.numel()/2)
        num_inst = int(placedb.num_movable_nodes_fence_region[region_id])
        final_movVal = np.zeros(2, dtype=np.float32).tolist()

        if region_id == 2:
            mask = model.data_collections.dsp_mask
            sites = placedb.dspSiteXYs
        elif region_id == 3:
            mask = model.data_collections.bram_mask
            sites = placedb.bramSiteXYs
        elif region_id == 4:
            mask = model.data_collections.uram_mask
            sites = placedb.uramSiteXYs

        available_sites = np.ones(len(sites), dtype=bool)
        rem_insts = np.ones(placedb.num_physical_nodes, dtype=bool)
        final_locX = np.ones(placedb.num_physical_nodes, dtype=np.float32)
        final_locX *= -1
        final_locY = np.ones_like(final_locX)
        final_locY *= -1

        #TODO-Include cascade shape handling
        #Handle region constraints if any
        for el in range(placedb.num_region_constraint_boxes):

            region_x_mask = np.logical_and(sites[:,0] >= placedb.region_box2xl[el], sites[:,0] < placedb.region_box2xh[el])
            region_y_mask = np.logical_and(sites[:,1] >= placedb.region_box2yl[el], sites[:,1] < placedb.region_box2yh[el])
            region_site_mask = np.logical_and(region_x_mask, region_y_mask)
            region_site_mask = np.logical_and(region_site_mask, available_sites == 1)

            region_instance_mask = np.logical_and(placedb.node2regionBox_map == el, placedb.node2fence_region_map == region_id)
            inst_mask = torch.from_numpy(region_instance_mask).to(pos.device)
            num_rInsts = region_instance_mask.sum()

            if num_rInsts > 0:
                rlocX = pos[:placedb.num_physical_nodes][inst_mask].cpu().detach().numpy()
                rlocY = pos[numNodes:numNodes+placedb.num_physical_nodes][inst_mask].cpu().detach().numpy()

                rSites = sites[region_site_mask]
                num_sites = len(rSites)
                precondWL = model.precondWL[:placedb.num_physical_nodes][inst_mask].cpu().detach().numpy()
                movVal = np.zeros(2, dtype=np.float32).tolist()
                outLoc = np.zeros(2*num_rInsts, dtype=np.float32).tolist()

                legalize_cpp.legalize(rlocX, rlocY, num_rInsts, num_sites, rSites.flatten(), precondWL, lg_max_dist_init, lg_max_dist_incr, lg_flow_cost_scale, movVal, outLoc)

                final_movVal[0] = max(movVal[0], final_movVal[0])
                final_movVal[1] += movVal[1]*num_rInsts
                
                final_locX[region_instance_mask] = outLoc[:num_rInsts]
                final_locY[region_instance_mask] = outLoc[num_rInsts:]

                rem_insts[region_instance_mask] = False
                occupied_sites = np.column_stack((np.array(outLoc[:num_rInsts]), np.array(outLoc[num_rInsts:])))

                #Do not consider occupied sites
                ##TODO - Improve to faster version
                for sEl in occupied_sites:
                    index = np.intersect1d(np.where(sites[:,0] == sEl[0])[0], np.where(sites[:,1] == sEl[1])[0])[0]
                    available_sites[index] = False

        #Handle remaining instances
        rem_insts_mask = np.logical_and(rem_insts, placedb.node2fence_region_map == region_id)
        num_rInsts = rem_insts_mask.sum()

        if num_rInsts > 0:
            rInst_mask = torch.from_numpy(rem_insts_mask).to(pos.device)
            rem_sites_mask = available_sites == 1
            rSites = sites[rem_sites_mask]
            num_sites = len(rSites)

            locX = pos[:placedb.num_physical_nodes][rInst_mask].cpu().detach().numpy()
            locY = pos[numNodes:numNodes+placedb.num_physical_nodes][rInst_mask].cpu().detach().numpy()

            precondWL = model.precondWL[:placedb.num_physical_nodes][rInst_mask].cpu().detach().numpy()
            movVal = np.zeros(2, dtype=np.float32).tolist()
            outLoc = np.zeros(2*num_rInsts, dtype=np.float32).tolist()
            
            legalize_cpp.legalize(locX, locY, num_rInsts, num_sites, rSites.flatten(), precondWL, lg_max_dist_init, lg_max_dist_incr, lg_flow_cost_scale, movVal, outLoc)

            final_movVal[0] = max(movVal[0], final_movVal[0])
            final_movVal[1] += movVal[1]*num_rInsts

            final_locX[rem_insts_mask] = outLoc[:num_rInsts]
            final_locY[rem_insts_mask] = outLoc[num_rInsts:]

        updLocX = torch.from_numpy(final_locX).to(dtype=pos.dtype, device=pos.device)
        updLocY = torch.from_numpy(final_locY).to(dtype=pos.dtype, device=pos.device)
        pos.data[:placedb.num_physical_nodes].masked_scatter_(mask, updLocX[mask])
        pos.data[numNodes:numNodes+placedb.num_physical_nodes].masked_scatter_(mask, updLocY[mask])

        final_movVal[1] /= num_inst

        return final_movVal 

