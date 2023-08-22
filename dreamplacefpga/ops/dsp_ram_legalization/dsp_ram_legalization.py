##
# @file   dsp_ram_legalization.py
# @author Rachel Selina (DREAMPlaceFPGA-PL) 
# @date   Oct 2020
#

import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np
import pdb 

import dreamplacefpga.ops.dsp_ram_legalization.dsp_ram_legalization_cpp as dsp_ram_legalization_cpp
import dreamplacefpga.configure as configure

import logging
logger = logging.getLogger(__name__)

class LegalizeDSPRAM(nn.Module):
    def __init__(self, data_collections, placedb, device):
        """
        @brief initialization
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        super(LegalizeDSPRAM, self).__init__()
        
        self.spiral_accessor = data_collections.spiral_accessor
        self.site_xy = data_collections.lg_siteXYs
        self.regionBox2xl = data_collections.regionBox2xl
        self.regionBox2yl = data_collections.regionBox2yl
        self.regionBox2xh = data_collections.regionBox2xh
        self.regionBox2yh = data_collections.regionBox2yh
        self.site_type_map = data_collections.site_type_map
        self.node2regionBox_map = data_collections.node2regionBox_map
        self.dsp_mask = data_collections.dsp_mask
        self.bram_mask = data_collections.bram_mask
        self.uram_mask = data_collections.uram_mask

        self.placedb = placedb
        self.num_movable_nodes_fence_region = placedb.num_movable_nodes_fence_region
        self.node2fence_region_map = placedb.node2fence_region_map
        self.node_size_x = placedb.node_size_x
        self.node_size_y = placedb.node_size_y
        self.resource_size_x = placedb.resource_size_x
        self.resource_size_y = placedb.resource_size_y
        self.dspSiteXYs = placedb.dspSiteXYs
        self.bramSiteXYs = placedb.bramSiteXYs
        self.uramSiteXYs = placedb.uramSiteXYs
        self.region_box2xl = placedb.region_box2xl
        self.region_box2yl = placedb.region_box2yl
        self.region_box2xh = placedb.region_box2xh
        self.region_box2yh = placedb.region_box2yh

        self.xl = placedb.xl
        self.yl = placedb.yl
        self.xh = placedb.xh
        self.yh = placedb.yh
        self.num_sites_x = placedb.num_sites_x
        self.num_sites_y = placedb.num_sites_y
        self.numNodes = placedb.num_nodes
        self.num_physical_nodes = placedb.num_physical_nodes
        self.num_region_constraint_boxes = placedb.num_region_constraint_boxes
        self.device = device

        #Constants
        self.lg_max_dist_init=10.0
        self.lg_max_dist_incr=10.0
        self.lg_flow_cost_scale=100.0
        self.spiralBegin = 0
        self.spiralEnd = self.spiral_accessor.shape[0]

    def legalize(self, pos, region_id, model):
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
        num_inst = int(self.num_movable_nodes_fence_region[region_id])
        final_movVal = np.zeros(2, dtype=np.float32).tolist()

        if region_id == 2:
            mask = self.dsp_mask
            sites = self.dspSiteXYs
        elif region_id == 3:
            mask = self.bram_mask
            sites = self.bramSiteXYs
        elif region_id == 4:
            mask = self.uram_mask
            sites = self.uramSiteXYs

        available_sites = np.ones(len(sites), dtype=bool)
        rem_insts = np.ones(self.num_physical_nodes, dtype=bool)
        final_locX = np.ones(self.num_physical_nodes, dtype=np.float32)
        final_locX *= -1
        final_locY = np.ones_like(final_locX)
        final_locY *= -1

        insts_mask = np.logical_and(rem_insts, self.node2fence_region_map == region_id)
        nodeSizeX = self.node_size_x[:self.num_physical_nodes]
        nodeSizeY = self.node_size_y[:self.num_physical_nodes]

        cascade_insts_mask = np.logical_and(insts_mask,
                    np.logical_or(nodeSizeX > self.resource_size_x[region_id],
                                  nodeSizeY > self.resource_size_y[region_id]))

        num_cascade_insts = cascade_insts_mask.sum()

        ##Handle cascade instances first along with region constraints if any
        if num_cascade_insts > 0:
            cascadeInsts_indices = np.where(cascade_insts_mask == True)[0]
            cascade_insts_index = torch.from_numpy(cascadeInsts_indices).to(dtype=torch.int, device=self.device)
            cascade_insts_spread = (self.node_size_y[cascadeInsts_indices]/self.resource_size_y[region_id]).astype(np.int32)
            cascade_sites = torch.from_numpy(cascade_insts_spread).to(self.device)
            cascade_sites_cumsum = torch.cumsum(cascade_sites, dim=0, dtype=torch.int)
            movVal = torch.zeros(2, dtype=pos.dtype, device=self.device)
            outLocX = torch.zeros(cascade_insts_spread.sum(), dtype=pos.dtype, device=self.device)
            outLocY = torch.zeros_like(outLocX)
            #Use grid map based on resource size
            x_grids = np.ceil(self.num_sites_x/self.resource_size_x[region_id]).astype(int)
            y_grids = np.ceil(self.num_sites_y/self.resource_size_y[region_id]).astype(int)
            resource_grid = torch.zeros((x_grids, y_grids), dtype=torch.int, device=self.device)

            if pos.is_cuda:
                cpu_movVal = movVal.cpu()
                cpu_outLocX = outLocX.cpu()
                cpu_outLocY = outLocY.cpu()
                
                dsp_ram_legalization_cpp.legalizeCascadeInsts(pos.cpu(), torch.flatten(self.site_xy).cpu(), self.regionBox2xl.cpu(),
                            self.regionBox2yl.cpu(), self.regionBox2xh.cpu(), self.regionBox2yh.cpu(),
                            model.data_collections.resource_size_x.cpu(), model.data_collections.resource_size_y.cpu(),
                            cascade_sites.cpu(), torch.flatten(self.spiral_accessor).cpu(), torch.flatten(self.site_type_map).cpu(),
                            self.node2regionBox_map.cpu(), cascade_insts_index.cpu(), cascade_sites_cumsum.cpu(),
                            self.xl, self.yl, self.xh, self.yh, region_id, self.spiralBegin, self.spiralEnd,
                            num_cascade_insts, self.num_sites_x, self.num_sites_y, cpu_movVal,
                            cpu_outLocX, cpu_outLocY)
                movVal.data.copy_(cpu_movVal)
                outLocX.data.copy_(cpu_outLocX)
                outLocY.data.copy_(cpu_outLocY)
            else:
                dsp_ram_legalization_cpp.legalizeCascadeInsts(pos, torch.flatten(self.site_xy), self.regionBox2xl,
                            self.regionBox2yl, self.regionBox2xh, self.regionBox2yh, model.data_collections.resource_size_x,
                            model.data_collections.resource_size_y, cascade_sites, torch.flatten(self.spiral_accessor),
                            torch.flatten(self.site_type_map), self.node2regionBox_map, cascade_insts_index,
                            cascade_sites_cumsum, self.xl, self.yl, self.xh, self.yh, region_id, self.spiralBegin,
                            self.spiralEnd, num_cascade_insts, self.num_sites_x, self.num_sites_y, movVal,
                            outLocX, outLocY)

            outLocX = outLocX.cpu().detach().numpy()
            outLocY = outLocY.cpu().detach().numpy()
            cascade_sites_cumsum = cascade_sites_cumsum.cpu().detach().numpy()
            cascade_sites_cumsum = np.insert(cascade_sites_cumsum, 0, 0)

            final_movVal[0] = max(movVal[0], final_movVal[0])
            final_movVal[1] += movVal[1]

            final_locX[cascadeInsts_indices] = outLocX[cascade_sites_cumsum[:-1]]
            final_locY[cascadeInsts_indices] = outLocY[cascade_sites_cumsum[:-1]]

            rem_insts[cascadeInsts_indices] = False

            occupied_sites = np.column_stack((outLocX, outLocY))
            #Do not consider occupied sites
            ##TODO - Improve to faster version
            for sEl in occupied_sites:
                index = np.intersect1d(np.where(sites[:,0] == sEl[0])[0], np.where(sites[:,1] == sEl[1])[0])[0]
                available_sites[index] = False

        #Handle region constraints if any in remaining instances
        available_region_sites = np.zeros(self.num_region_constraint_boxes, dtype=np.int32)
        insts_in_region = np.zeros_like(available_region_sites)

        all_site_masks = []
        all_insts_masks = []
        for el in range(self.num_region_constraint_boxes):

            region_x_mask = np.logical_and(sites[:,0] >= self.region_box2xl[el], sites[:,0] < self.region_box2xh[el])
            region_y_mask = np.logical_and(sites[:,1] >= self.region_box2yl[el], sites[:,1] < self.region_box2yh[el])
            region_site_mask = np.logical_and(region_x_mask, region_y_mask)
            all_site_masks.append(region_site_mask)

            rSites = sites[region_site_mask]
            available_region_sites[el] = len(rSites)

            region_instance_mask = np.logical_and(self.placedb.node2regionBox_map == el, np.logical_and(rem_insts, self.node2fence_region_map == region_id))
            all_insts_masks.append(region_instance_mask)
            insts_in_region[el] = region_instance_mask.sum()

        all_site_masks = np.array(all_site_masks)
        all_insts_masks = np.array(all_insts_masks)

        #Start with smallest region
        region_index_ordering = np.argsort(available_region_sites)
        for idx in range(self.num_region_constraint_boxes):
            el = region_index_ordering[idx]

            region_site_mask = np.logical_and(all_site_masks[el], available_sites == 1)
            region_instance_mask = all_insts_masks[el]

            inst_mask = torch.from_numpy(region_instance_mask).to(self.device)
            num_rInsts = region_instance_mask.sum()

            if num_rInsts > 0:
                rlocX = pos[:self.num_physical_nodes][inst_mask].cpu().detach().numpy()
                rlocY = pos[self.numNodes:self.numNodes+self.num_physical_nodes][inst_mask].cpu().detach().numpy()

                rSites = sites[region_site_mask]
                num_sites = len(rSites)
                precondWL = model.precondWL[:self.num_physical_nodes][inst_mask].cpu().detach().numpy()
                movVal = np.zeros(2, dtype=np.float32).tolist()
                outLoc = np.zeros(2*num_rInsts, dtype=np.float32).tolist()

                #Provide message if there are not enough sites for the instances to be legalized
                if num_rInsts > num_sites:
                    print("WARNING: Number of sites: %d and number of insts: %d"%(num_sites, num_rInsts))

                dsp_ram_legalization_cpp.legalize(rlocX, rlocY, num_rInsts, num_sites, rSites.flatten(), precondWL, self.lg_max_dist_init,
                                      self.lg_max_dist_incr, self.lg_flow_cost_scale, movVal, outLoc)

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
        rem_insts_mask = np.logical_and(rem_insts, self.node2fence_region_map == region_id)
        num_rInsts = rem_insts_mask.sum()

        if num_rInsts > 0:
            rInst_mask = torch.from_numpy(rem_insts_mask).to(self.device)
            rem_sites_mask = available_sites == 1
            rSites = sites[rem_sites_mask]
            num_sites = len(rSites)

            locX = pos[:self.num_physical_nodes][rInst_mask].cpu().detach().numpy()
            locY = pos[self.numNodes:self.numNodes+self.num_physical_nodes][rInst_mask].cpu().detach().numpy()

            precondWL = model.precondWL[:self.num_physical_nodes][rInst_mask].cpu().detach().numpy()
            movVal = np.zeros(2, dtype=np.float32).tolist()
            outLoc = np.zeros(2*num_rInsts, dtype=np.float32).tolist()
            
            dsp_ram_legalization_cpp.legalize(locX, locY, num_rInsts, num_sites, rSites.flatten(), precondWL, self.lg_max_dist_init,
                                  self.lg_max_dist_incr, self.lg_flow_cost_scale, movVal, outLoc)

            final_movVal[0] = max(movVal[0], final_movVal[0])
            final_movVal[1] += movVal[1]*num_rInsts

            final_locX[rem_insts_mask] = outLoc[:num_rInsts]
            final_locY[rem_insts_mask] = outLoc[num_rInsts:]

        updLocX = torch.from_numpy(final_locX).to(dtype=pos.dtype, device=self.device)
        updLocY = torch.from_numpy(final_locY).to(dtype=pos.dtype, device=self.device)
        pos.data[:self.num_physical_nodes].masked_scatter_(mask, updLocX[mask])
        pos.data[self.numNodes:self.numNodes+self.num_physical_nodes].masked_scatter_(mask, updLocY[mask])

        final_movVal[1] /= num_inst
        
        #Checker
        final_sites = final_locX[self.node2fence_region_map == region_id]*self.num_sites_y + final_locY[self.node2fence_region_map == region_id]
        if final_sites.shape[0] != np.unique(final_sites).shape[0]:
            print("ERROR: Multiple instances %d of type %d occupy the same sites - CHECK" % (final_sites.shape[0]-np.unique(final_sites).shape[0], region_id))
        #Checker

        return final_movVal 

