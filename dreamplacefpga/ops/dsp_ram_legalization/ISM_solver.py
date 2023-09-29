##
# @file   ISM_solver.py
# @author Zhili Xiong (DREAMPlaceFPGA-Macro) 
# @date   Sep 2023
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

class ISMSolver(nn.Module):
    def __init__(self, data_collections, placedb, device):
        """
        @brief initialization
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        super(ISMSolver, self).__init__()
        
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
        self.net_mask = data_collections.net_mask_ISM

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
        self.Rmax = 50
        self.Nmax = 50
        self.Uth = 0.7

        # Variables
        self.region_id = None
        self.sites = None
        self.route_utilization_map = None
        self.insts_indices = []
        self.cascade_insts_indices = []
        self.independent_sets = []
        self.inst_site_masks = []
        self.inst_connection = {}
        self.inst_dep = {}
        self.conn_nets = []
        self.site2inst_map = {}
        self.inst2site_map = {}
        self.net_bboxes = {}
        self.res_hpwl = {}
        self.cost_matrix = {}

    def build_site2inst_map(self, pos):
        """
        @brief build site2inst map
        """
        locX = pos[:self.num_physical_nodes].cpu().detach().numpy()
        locY = pos[self.numNodes:self.numNodes+self.num_physical_nodes].cpu().detach().numpy()

        site2inst_map = {i: -1 for i in range(len(self.sites))} # site_id: inst_id, -1 means no inst
        inst2site_map = {i: -1 for i in self.insts_indices} # inst_id: site_id, -1 means no site 
        for inst_id in self.insts_indices:
            sEl = np.intersect1d(np.where(self.sites[:,0] == locX[inst_id])[0], np.where(self.sites[:,1] == locY[inst_id])[0])[0]
            site2inst_map[sEl] = inst_id
            inst2site_map[inst_id] = sEl

        # update empty sitess
        for cascade_inst_id in self.cascade_insts_indices:
            sEl = np.intersect1d(np.where(self.sites[:,0] == locX[cascade_inst_id])[0], np.where(self.sites[:,1] == locY[cascade_inst_id])[0])[0]
            cascade_inst_spread = (self.node_size_y[cascade_inst_id] / self.resource_size_y[self.region_id]).astype(np.int32) 
            for i in range(cascade_inst_spread):
                site2inst_map[sEl+i] = cascade_inst_id
                inst2site_map[cascade_inst_id] = sEl+i
                
        return site2inst_map, inst2site_map
    
    
    def init_ISM(self, pos, region_id, model, route_utilization_map):
        """
        @brief initialize ISM solver
        """

        if region_id == 2:
            mask = self.dsp_mask
            self.sites = self.dspSiteXYs
        elif region_id == 3:
            mask = self.bram_mask
            self.sites = self.bramSiteXYs
        elif region_id == 4:
            mask = self.uram_mask
            self.sites = self.uramSiteXYs
        
        self.region_id = region_id

        num_inst = int(self.num_movable_nodes_fence_region[region_id])
        rem_insts = np.ones(self.num_physical_nodes, dtype=bool)

        insts_mask = np.logical_and(rem_insts, self.node2fence_region_map == region_id)

        nodeSizeX = self.node_size_x[:self.num_physical_nodes]
        nodeSizeY = self.node_size_y[:self.num_physical_nodes]
        cascade_insts_mask = np.logical_and(insts_mask,
                    np.logical_or(nodeSizeX > self.resource_size_x[region_id],
                                  nodeSizeY > self.resource_size_y[region_id]))

        num_cascade_insts = cascade_insts_mask.sum()

        # Update remaining instances without cascade instances
        num_inst = num_inst - num_cascade_insts
        insts_mask = np.logical_and(insts_mask, np.logical_not(cascade_insts_mask))
        self.insts_indices = np.where(insts_mask == True)[0]
        self.cascade_insts_indices = np.where(cascade_insts_mask == True)[0]

        # build site2inst map
        self.site2inst_map, self.inst2site_map = self.build_site2inst_map(pos)

        self.route_utilization_map = route_utilization_map

        # build inst2inst connection
        for inst_id in self.insts_indices:
            self.inst_connection[inst_id] = []
            self.inst_dep[inst_id] = 0 # 0 means no dependency
            for pin_id in self.placedb.node2pin_map[inst_id]:
                net_id = self.placedb.pin2net_map[pin_id]
                if self.net_mask[net_id]:
                    for k in self.placedb.net2pin_map[net_id]:
                        node_id = self.placedb.pin2node_map[k]
                        if k != pin_id and node_id not in self.inst_connection[inst_id] and node_id in self.insts_indices:
                            self.inst_connection[inst_id].append(node_id)

                    if net_id not in self.conn_nets:
                        self.conn_nets.append(net_id)

        self.build_net_bbox(pos)

    def build_indep_sets(self):
        """
        @brief build independent sets
        """
        for inst_id in self.insts_indices:

            if self.inst_dep[inst_id] == 1:
                continue

            instLimit_xl, instLimit_yl, instLimit_xh, instLimit_yh = self.xl, self.yl, self.xh, self.yh
            # build instLimit for inst in region
            if self.node2regionBox_map[inst_id] != -1:
                box_id = self.node2regionBox_map[inst_id]
                instLimit_xl = self.regionBox2xl[box_id]
                instLimit_yl = self.regionBox2yl[box_id]
                instLimit_xh = self.regionBox2xh[box_id]
                instLimit_yh = self.regionBox2yh[box_id]

            indep_set = []
            inst_site_mask = []
            seed = inst_id
            indep_set.append(seed)
            inst_site_mask.append(1) # 1 means inst, 0 means empty site
            self.inst_dep[seed] = 1

            sEl = self.inst2site_map[seed]

            stepSize = self.resource_size_y[self.region_id]
            sElX = sEl // (self.num_sites_y/stepSize)
            sElY = sEl % (self.num_sites_y/stepSize)

            spiral_accessor = torch.flatten(self.spiral_accessor).cpu().detach().numpy()

            for sId in range(0, self.Rmax):
                xindex = sElX + spiral_accessor[2*sId]
                yindex = sElY + spiral_accessor[2*sId+1]
  
                index = int(xindex * (self.num_sites_y/stepSize) + yindex)

                if index < 0 or index >= len(self.sites):
                    continue

                xval = int(self.sites[index][0])
                yval = int(self.sites[index][1])

                if xval < instLimit_xl or xval > instLimit_xh or yval < instLimit_yl or yval > instLimit_yh or self.site_type_map[xval][yval] != self.region_id:
                    continue
                
                nbr = self.site2inst_map[index]

                # empty sites
                if nbr == -1:
                    indep_set.append(index)
                    inst_site_mask.append(0)

                if nbr in self.inst_connection[seed]:
                    continue

                if nbr in self.insts_indices and self.inst_dep[nbr] == 0:
                    indep_set.append(nbr)
                    inst_site_mask.append(1)
                    self.inst_dep[nbr] = 1
                    for conn in self.inst_connection[nbr]:
                        self.inst_dep[conn] = 1

                if len(indep_set) >= self.Nmax:
                    break

            self.independent_sets.append(indep_set)
            self.inst_site_masks.append(inst_site_mask)


    def run_ISM(self, pos, region_id, model, route_utilization_map):
        """
        @brief run ISM solver
        """
        self.init_ISM(pos, region_id, model, route_utilization_map)
        self.build_indep_sets()

        for indep_set, inst_site_mask in zip(self.independent_sets, self.inst_site_masks):
            mem_bboxes, mem_offsets, mem_net_ids, mem_ranges = self.compute_net_bbox(pos, indep_set, inst_site_mask)
            cost_matrix = self.compute_cost_matrix(indep_set, inst_site_mask, mem_bboxes, mem_offsets, mem_net_ids, mem_ranges)
            pdb.set_trace()


    def build_net_bbox(self, pos):
        """
        @brief build net bounding box
        """
        locX = pos[:self.num_physical_nodes].cpu().detach().numpy()
        locY = pos[self.numNodes:self.numNodes+self.num_physical_nodes].cpu().detach().numpy()

        self.net_bboxes = {net_id: (self.xh, self.yh, self.xl, self.yl) for net_id in self.conn_nets}
        self.res_hpwl = {net_id: 0 for net_id in self.conn_nets}
        for net_id in self.conn_nets:
            for pin_id in self.placedb.net2pin_map[net_id]:
                node_id = self.placedb.pin2node_map[pin_id]
                xlo = min(self.net_bboxes[net_id][0], locX[node_id]+self.placedb.pin_offset_x[pin_id])
                ylo = min(self.net_bboxes[net_id][1], locY[node_id]+self.placedb.pin_offset_y[pin_id])
                xhi = max(self.net_bboxes[net_id][2], locX[node_id]+self.placedb.pin_offset_x[pin_id])
                yhi = max(self.net_bboxes[net_id][3], locY[node_id]+self.placedb.pin_offset_y[pin_id])
                self.net_bboxes[net_id] = (xlo, ylo, xhi, yhi)  

            self.res_hpwl[net_id] = (self.net_bboxes[net_id][2] - self.net_bboxes[net_id][0]) + (self.net_bboxes[net_id][3] - self.net_bboxes[net_id][1])

        sum_hpwl = sum(self.res_hpwl.values())
        print("intial hpwl: ", sum_hpwl)

    def compute_net_bbox(self, pos, indep_set, inst_site_mask):
        """
        @brief compute net bounding box
        """
        locX = pos[:self.num_physical_nodes].cpu().detach().numpy()
        locY = pos[self.numNodes:self.numNodes+self.num_physical_nodes].cpu().detach().numpy()

        mem_bboxes = []
        mem_offsets = []
        mem_net_ids = []
        mem_ranges = []
        mem_ranges.append(0)

        for i, inst in enumerate(indep_set):
            if inst_site_mask[i] == 0:
                mem_ranges.append(len(mem_bboxes))
                continue
            
            for pin_id in self.placedb.node2pin_map[inst]:
                net_id = self.placedb.pin2net_map[pin_id]
                if self.net_mask[net_id] == 0:
                    continue

                mem_bboxes.append(self.net_bboxes[net_id])
                mem_offsets.append((self.placedb.pin_offset_x[pin_id], self.placedb.pin_offset_y[pin_id]))
                mem_net_ids.append(net_id)
            
            mem_ranges.append(len(mem_bboxes))

        return mem_bboxes, mem_offsets, mem_net_ids, mem_ranges


    def compute_cost_matrix(self, indep_set, inst_site_mask, mem_bboxes, mem_offsets, mem_net_ids, mem_ranges):
        """
        @brief build cost matrix
        """
        cost_matrix = np.zeros((len(indep_set), len(indep_set)))
        for i, inst_site in enumerate(indep_set):
            for j, inst_site in enumerate(indep_set):
                cost = 0
                if inst_site_mask[j] == 1:
                    site = self.inst2site_map[inst_site]
                else:
                    site = inst_site

                for k in range(mem_ranges[i], mem_ranges[i+1]):
                    locx = self.sites[site][0] + mem_offsets[k][0]
                    locy = self.sites[site][1] + mem_offsets[k][1]
                    cost += max(0, mem_bboxes[k][0] - locx, locx - mem_bboxes[k][2]) + max(0, mem_bboxes[k][1] - locy, locy - mem_bboxes[k][3])
                    print(" the cost of moving instance %d from (%d, %d) to (%d, %d) is %d" % (inst_site, mem_bboxes[k][0], mem_bboxes[k][1], locx, locy, cost))

                cost_matrix[i][j] = cost * self.lg_flow_cost_scale 

            # add congestion cost
            if self.route_utilization_map[int(self.sites[site][0]), int(self.sites[site][1])] > self.Uth:
                cost_matrix[i][j] += np.inf

        return cost_matrix





        

