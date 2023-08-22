##
# @file   lut_ff_legalization.py
# @author Rachel Selina (DREAMPlaceFPGA-PL) 
# @date   Apr 2022
#

import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb 
import time
import logging

import dreamplacefpga.ops.lut_ff_legalization.lut_ff_legalization_cpp as lut_ff_legalization_cpp
import dreamplacefpga.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplacefpga.ops.lut_ff_legalization.lut_ff_legalization_cuda as lut_ff_legalization_cuda

class LegalizeCLB(nn.Module):
    def __init__(self, placedb, data_collections, net_wts, avg_lut_area, avg_ff_area, inst_areas, site_types, 
        num_threads, device):

        super(LegalizeCLB, self).__init__()

        self.lut_flop_indices=data_collections.flop_lut_indices
        self.num_lutflops=self.lut_flop_indices.shape[0]
        self.num_nodes=placedb.num_physical_nodes

        #Use nodeNames for debug purpose - Convert string to int
        self.nodeNames=torch.arange(self.num_nodes, dtype=torch.int, device=device)

        self.flop2ctrlSetId_map=data_collections.flop2ctrlSetId_map
        self.flop_ctrlSets=data_collections.flop_ctrlSets
        self.pin2node_map=data_collections.pin2node_map
        self.pin2net_map=data_collections.pin2net_map
        self.flat_net2pin_map=data_collections.flat_net2pin_map
        self.flat_net2pin_start_map=data_collections.flat_net2pin_start_map
        self.flat_node2pin_map=data_collections.flat_node2pin_map
        self.flat_node2pin_start_map=data_collections.flat_node2pin_start_map
        self.node2fence_region_map=data_collections.node2fence_region_map
        self.node2outpinIdx_map=data_collections.node2outpinIdx_map[:placedb.num_physical_nodes]
        self.pin_typeIds=data_collections.pin_typeIds
        self.lut_type=data_collections.lut_type
        self.lut_type[self.node2fence_region_map == 0] += 1
        self.net_wts=net_wts
        self.avg_lut_area=avg_lut_area
        self.avg_ff_area=avg_ff_area
        self.inst_areas=inst_areas
        self.pin_offset_x=data_collections.lg_pin_offset_x
        self.pin_offset_y=data_collections.lg_pin_offset_y
        self.site_types=site_types
        self.site_xy=data_collections.lg_siteXYs
        self.node_size_x=data_collections.node_size_x[:placedb.num_physical_nodes]
        self.node_size_y=data_collections.node_size_y[:placedb.num_physical_nodes]
        self.net2pincount=data_collections.net2pincount_map
        self.node2pincount=data_collections.node2pincount_map
        self.spiral_accessor=data_collections.spiral_accessor
        self.num_nets=placedb.num_nets
        self.num_movable_nodes=placedb.num_movable_nodes
        self.num_sites_x=placedb.num_sites_x
        self.num_sites_y=placedb.num_sites_y
        self.xl=placedb.xl
        self.yl=placedb.yl
        self.xh=placedb.xh
        self.yh=placedb.yh
        self.xWirelenWt=placedb.xWirelenWt
        self.yWirelenWt=placedb.yWirelenWt
        self.nbrDistEnd=placedb.nbrDistEnd
        self.num_threads=num_threads
        self.device=device

        #Region Constraints
        self.regionBox2xl = data_collections.regionBox2xl
        self.regionBox2yl = data_collections.regionBox2yl
        self.regionBox2xh = data_collections.regionBox2xh
        self.regionBox2yh = data_collections.regionBox2yh
        self.node2regionBox_map = data_collections.node2regionBox_map
        self.is_macro_inst = data_collections.is_macro_inst

        #Initialize required constants
        self.int_min_val = -2147483647
        self.maxList = math.ceil(0.005 * self.num_nodes) #Based on empirical results from elfPlace
        self.nbrDistBeg = 1.0
        self.nbrDistIncr = 1.0
        self.numGroups = math.ceil((self.nbrDistEnd-self.nbrDistBeg)/self.nbrDistIncr) + 1
        self.WLscoreMaxNetDegree = 100
        self.extNetCountWt = 0.3
        self.wirelenImprovWt = 0.1

        #Architecture specific values
        self.CKSR_IN_CLB = 2
        self.CE_IN_CLB = 4
        self.netShareScoreMaxNetDegree = 16
        self.SLICE_CAPACITY = 16
        self.HALF_SLICE_CAPACITY = int(self.SLICE_CAPACITY/2)
        self.BLE_CAPACITY = 2
        self.NUM_BLE_PER_SLICE = int(self.SLICE_CAPACITY/self.BLE_CAPACITY)
        self.NUM_BLE_PER_HALF_SLICE = int(self.HALF_SLICE_CAPACITY/self.BLE_CAPACITY)

        self.PQ_IDX = 10
        self.SCL_IDX = 128 #2*self.SLICE_CAPACITY
        self.SIG_IDX = 2*self.SLICE_CAPACITY #32
        #Initialize required tensors
        self.net_bbox = torch.zeros(self.num_nets*4, dtype=self.node_size_x.dtype, device=device)

        self.net_pinIdArrayX = torch.zeros(len(self.flat_net2pin_map), dtype=torch.int, device=device)
        self.net_pinIdArrayY = torch.zeros_like(self.net_pinIdArrayX) #len(self.flat_net2pin_map)

        self.flat_node2precluster_map = torch.ones((self.num_nodes,3), dtype=torch.int, device=device)
        self.flat_node2precluster_map *= -1
        self.flat_node2precluster_map[:,0] = torch.arange(self.num_nodes, dtype=torch.int, device=device)
        self.flat_node2prclstrCount = torch.ones(self.num_nodes, dtype=torch.int, device=device)

        #Instance Candidates
        self.inst_curr_detSite = torch.zeros_like(self.flat_node2prclstrCount) #num_nodes
        self.inst_curr_detSite[self.node2fence_region_map < 2] = -1
        self.inst_curr_bestSite = torch.zeros_like(self.inst_curr_detSite) #num_nodes
        self.inst_curr_bestSite[self.node2fence_region_map < 2] = -1
        self.inst_curr_bestScoreImprov = torch.zeros(self.num_nodes, dtype=self.node_size_x.dtype, device=device)
        self.inst_curr_bestScoreImprov[self.node2fence_region_map < 2] = -10000.0

        self.inst_next_detSite = torch.zeros_like(self.inst_curr_detSite) #num_nodes
        self.inst_next_detSite[self.node2fence_region_map < 2] = -1
        self.inst_next_bestSite = torch.zeros_like(self.inst_next_detSite) #num_nodes
        self.inst_next_bestSite[self.node2fence_region_map < 2] = -1
        self.inst_next_bestScoreImprov = torch.zeros_like(self.inst_curr_bestScoreImprov) #num_nodes
        self.inst_next_bestScoreImprov[self.node2fence_region_map < 2] = -10000.0

        self.num_clb_sites = torch.bincount(self.site_types.flatten())[1].item()
        #Map from mem addr to CLB site
        self.addr2site_map = self.site_types.flatten().nonzero(as_tuple=True)[0]
        #Map from CLB site to mem addr
        self.site2addr_map = torch.ones(self.num_sites_x*self.num_sites_y, dtype=torch.int, device=device)
        self.site2addr_map *= -1
        self.site2addr_map[self.addr2site_map] = torch.arange(self.num_clb_sites, dtype=torch.int, device=device)
        self.addr2site_map = self.addr2site_map.int()

        #Site Neighbors
        self.site_nbrList = torch.zeros((self.num_clb_sites, self.maxList), dtype=torch.int, device=device)
        self.site_nbr = torch.zeros_like(self.site_nbrList) #num_clb_sites * maxList
        self.site_nbr_idx = torch.zeros(self.num_clb_sites, dtype=torch.int, device=device)
        self.site_nbrRanges = torch.zeros((self.num_clb_sites, self.numGroups+1), dtype=torch.int, device=device)
        self.site_nbrRanges_idx = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        self.site_nbrGroup_idx = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        ##Site Candidates
        self.site_det_score = torch.zeros(self.num_clb_sites, dtype=self.node_size_x.dtype, device=device)
        self.site_det_siteId = torch.ones_like(self.site_nbr_idx) #num_clb_sites
        self.site_det_siteId *= -1
        self.site_det_impl_lut = torch.ones((self.num_clb_sites, self.SLICE_CAPACITY), dtype=torch.int, device=device)
        self.site_det_impl_lut *= -1
        self.site_det_impl_ff = torch.ones_like(self.site_det_impl_lut) #num_clb_sites * SLICE_CAPACITY
        self.site_det_impl_ff *= -1
        self.site_det_impl_cksr = torch.ones((self.num_clb_sites, self.CKSR_IN_CLB), dtype=torch.int, device=device)
        self.site_det_impl_cksr *= -1
        self.site_det_impl_ce = torch.ones((self.num_clb_sites, self.CE_IN_CLB), dtype=torch.int, device=device)
        self.site_det_impl_ce *= -1
        self.site_det_sig = torch.zeros((self.num_clb_sites, self.SIG_IDX), dtype=torch.int, device=device)
        self.site_det_sig_idx = torch.zeros_like(self.site_det_siteId) #num_clb_sites

        self.site_curr_stable = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        self.site_curr_scl_idx = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        self.site_curr_scl_validIdx = torch.ones((self.num_clb_sites, self.SCL_IDX), dtype=torch.int, device=device)
        self.site_curr_scl_validIdx *= -1
        self.site_curr_scl_siteId = torch.ones((self.num_clb_sites, self.SCL_IDX), dtype=torch.int, device=device) #num_clb_sites * SCL_IDX
        self.site_curr_scl_siteId *= -1
        self.site_curr_scl_score = torch.zeros((self.num_clb_sites, self.SCL_IDX), dtype=self.node_size_x.dtype, device=device)
        self.site_curr_scl_impl_lut = torch.ones((self.num_clb_sites, self.SCL_IDX, self.SLICE_CAPACITY), dtype=torch.int, device=device)
        self.site_curr_scl_impl_lut *= -1
        self.site_curr_scl_impl_ff = torch.ones_like(self.site_curr_scl_impl_lut) #num_clb_sites * SCL_IDX * SLICE_CAPACITY
        self.site_curr_scl_impl_ff *= -1
        self.site_curr_scl_impl_cksr = torch.ones((self.num_clb_sites, self.SCL_IDX, self.CKSR_IN_CLB), dtype=torch.int, device=device)
        self.site_curr_scl_impl_cksr *= -1
        self.site_curr_scl_impl_ce = torch.ones((self.num_clb_sites, self.SCL_IDX, self.CE_IN_CLB), dtype=torch.int, device=device)
        self.site_curr_scl_impl_ce *= -1
        self.site_curr_scl_sig = torch.zeros((self.num_clb_sites, self.SCL_IDX, self.SIG_IDX), dtype=torch.int, device=device)
        self.site_curr_scl_sig_idx = torch.zeros_like(self.site_curr_scl_siteId) #num_clb_sites * SCL_IDX

        self.site_curr_pq_idx = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        self.site_curr_pq_top_idx = torch.ones_like(self.site_nbr_idx) #num_clb_sites
        self.site_curr_pq_top_idx *= -1
        self.site_curr_pq_score = torch.zeros((self.num_clb_sites, self.PQ_IDX), dtype=self.node_size_x.dtype, device=device)
        self.site_curr_pq_validIdx = torch.ones((self.num_clb_sites, self.PQ_IDX), dtype=torch.int, device=device)
        self.site_curr_pq_validIdx *= -1
        self.site_curr_pq_siteId = torch.ones((self.num_clb_sites, self.PQ_IDX), dtype=torch.int, device=device) #num_clb_sites * PQ_IDX
        self.site_curr_pq_siteId *= -1
        self.site_curr_pq_sig = torch.zeros((self.num_clb_sites, self.PQ_IDX, self.SIG_IDX), dtype=torch.int, device=device)
        self.site_curr_pq_sig_idx = torch.zeros_like(self.site_curr_pq_siteId) #num_clb_sites * PQ_IDX
        self.site_curr_pq_impl_lut = torch.ones((self.num_clb_sites, self.PQ_IDX, self.SLICE_CAPACITY), dtype=torch.int, device=device)
        self.site_curr_pq_impl_lut *= -1
        self.site_curr_pq_impl_ff = torch.ones_like(self.site_curr_pq_impl_lut) #num_clb_sites * PQ_IDX * SLICE_CAPACITY.
        self.site_curr_pq_impl_ff *= -1
        self.site_curr_pq_impl_cksr = torch.ones((self.num_clb_sites, self.PQ_IDX, self.CKSR_IN_CLB), dtype=torch.int, device=device)
        self.site_curr_pq_impl_cksr *= -1
        self.site_curr_pq_impl_ce = torch.ones((self.num_clb_sites, self.PQ_IDX, self.CE_IN_CLB), dtype=torch.int, device=device)
        self.site_curr_pq_impl_ce *= -1

        self.site_next_stable = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        self.site_next_scl_idx = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        self.site_next_scl_validIdx = torch.ones_like(self.site_curr_scl_validIdx) #num_clb_sites * SCL_IDX
        self.site_next_scl_validIdx *= -1
        self.site_next_scl_siteId = torch.ones_like(self.site_curr_scl_siteId) #num_clb_sites * SCL_IDX
        self.site_next_scl_siteId *= -1
        self.site_next_scl_score = torch.zeros_like(self.site_curr_scl_score) #num_clb_sites * SCL_IDX
        self.site_next_scl_impl_lut = torch.ones_like(self.site_curr_scl_impl_lut) #num_clb_sites * SCL_IDX * SLICE_CAPACITY
        self.site_next_scl_impl_lut *= -1
        self.site_next_scl_impl_ff = torch.ones_like(self.site_next_scl_impl_lut) #num_clb_sites * SCL_IDX * SLICE_CAPACITY
        self.site_next_scl_impl_ff *= -1
        self.site_next_scl_impl_cksr = torch.ones_like(self.site_curr_scl_impl_cksr) #num_clb_sites * SCL_IDX * CKSR_IN_CLB
        self.site_next_scl_impl_cksr *= -1
        self.site_next_scl_impl_ce = torch.ones_like(self.site_curr_scl_impl_ce) #num_clb_sites * SCL_IDX * CE_IN_CLB
        self.site_next_scl_impl_ce *= -1
        self.site_next_scl_sig = torch.zeros_like(self.site_curr_scl_sig) #num_clb_sites * SCL_IDX * SIG_IDX
        self.site_next_scl_sig_idx = torch.zeros_like(self.site_next_scl_siteId) #num_clb_sites * SCL_IDX

        self.site_next_pq_idx = torch.zeros_like(self.site_nbr_idx) #num_clb_sites
        self.site_next_pq_top_idx = torch.ones_like(self.site_nbr_idx) #num_clb_sites
        self.site_next_pq_top_idx *= -1
        self.site_next_pq_score = torch.zeros_like(self.site_curr_pq_score) #num_clb_sites * PQ_IDX
        self.site_next_pq_validIdx = torch.ones_like(self.site_curr_pq_validIdx) #num_clb_sites * PQ_IDX
        self.site_next_pq_validIdx *= -1
        self.site_next_pq_siteId = torch.ones_like(self.site_curr_pq_siteId) #num_clb_sites * PQ_IDX
        self.site_next_pq_siteId *= -1
        self.site_next_pq_sig = torch.zeros_like(self.site_curr_pq_sig) #num_clb_sites * PQ_IDX * SIG_IDX
        self.site_next_pq_sig_idx = torch.zeros_like(self.site_curr_pq_validIdx) #num_clb_sites * PQ_IDX
        self.site_next_pq_impl_lut = torch.ones_like(self.site_curr_pq_impl_lut) #num_clb_sites * PQ_IDX * SLICE_CAPACITY
        self.site_next_pq_impl_lut *= -1
        self.site_next_pq_impl_ff = torch.ones_like(self.site_next_pq_impl_lut) #num_clb_sites * PQ_IDX * SLICE_CAPACITY
        self.site_next_pq_impl_ff *= -1
        self.site_next_pq_impl_cksr = torch.ones_like(self.site_curr_pq_impl_cksr) #num_clb_sites * PQ_IDX * CKSR_IN_CLB
        self.site_next_pq_impl_cksr *= -1
        self.site_next_pq_impl_ce = torch.ones_like(self.site_curr_pq_impl_ce) #num_clb_sites * PQ_IDX * CE_IN_CLB
        self.site_next_pq_impl_ce *= -1

        self.inst_score_improv = torch.zeros(self.num_nodes, dtype=torch.int, device=self.device)
        self.inst_score_improv[self.node2fence_region_map < 2] = self.int_min_val
        self.site_score_improv = torch.zeros(self.num_clb_sites, dtype=torch.int, device=self.device)
        self.site_score_improv *= self.int_min_val

    def initialize(self, pos, wlPrecond, sorted_node_map, sorted_node_idx, sorted_net_map, sorted_net_idx, sorted_pin_map):
        tt = time.time()

        preClusteringMaxDist = 4.0
        maxD = math.ceil(self.nbrDistEnd) + 1
        spiralBegin = 0
        spiralEnd_maxD = 2 * (maxD + 1) * maxD + 1

        if pos.is_cuda:
            lut_ff_legalization_cuda.initLegalization(pos, self.pin_offset_x, self.pin_offset_y, 
                sorted_net_idx, sorted_node_map, sorted_node_idx, self.flat_net2pin_map, 
                self.flat_net2pin_start_map, self.flop2ctrlSetId_map, self.flop_ctrlSets, 
                self.node2fence_region_map, self.node2outpinIdx_map, self.pin2net_map, self.pin2node_map,
                self.pin_typeIds, self.net2pincount, self.num_nets, self.num_nodes,
                preClusteringMaxDist, self.WLscoreMaxNetDegree, self.net_bbox, self.net_pinIdArrayX,
                self.net_pinIdArrayY, self.flat_node2precluster_map, self.flat_node2prclstrCount)

            cpu_site_nbrList = torch.flatten(self.site_nbrList).cpu()
            cpu_site_nbrRanges = torch.flatten(self.site_nbrRanges).cpu()
            cpu_site_nbrRanges_idx = torch.flatten(self.site_nbrRanges_idx).cpu()
            cpu_site_nbr = torch.flatten(self.site_nbr).cpu()
            cpu_site_nbr_idx = torch.flatten(self.site_nbr_idx).cpu()
            cpu_site_nbrGroup_idx = torch.flatten(self.site_nbrGroup_idx).cpu()
            cpu_site_det_siteId = torch.flatten(self.site_det_siteId).cpu()
            cpu_site_curr_scl_siteId = torch.flatten(self.site_curr_scl_siteId).cpu()
            cpu_site_curr_scl_validIdx = torch.flatten(self.site_curr_scl_validIdx).cpu()
            cpu_site_curr_scl_idx = torch.flatten(self.site_curr_scl_idx).cpu()

            lut_ff_legalization_cpp.updateNbrListMap(
               pos.cpu(), wlPrecond.cpu(), torch.flatten(self.site_xy).cpu(), sorted_node_idx.cpu(), 
               self.node2fence_region_map.cpu(), torch.flatten(self.flat_node2precluster_map).cpu(), 
               self.flat_node2prclstrCount.cpu(), torch.flatten(self.site_types).cpu(), 
               torch.flatten(self.spiral_accessor).cpu(), self.site2addr_map.cpu(), self.addr2site_map.cpu(),
               self.num_nodes, self.num_sites_x, self.num_sites_y, self.num_clb_sites,
               self.SCL_IDX, spiralBegin, spiralEnd_maxD, self.maxList, self.nbrDistEnd, self.nbrDistBeg, 
               self.nbrDistIncr, self.numGroups, self.num_threads, 
               cpu_site_nbrList, cpu_site_nbrRanges, cpu_site_nbrRanges_idx, cpu_site_nbr, 
               cpu_site_nbr_idx, cpu_site_nbrGroup_idx, cpu_site_det_siteId, cpu_site_curr_scl_siteId, 
               cpu_site_curr_scl_validIdx, cpu_site_curr_scl_idx)

            torch.flatten(self.site_nbrList).data.copy_(cpu_site_nbrList.data)
            torch.flatten(self.site_nbrRanges).data.copy_(cpu_site_nbrRanges)
            torch.flatten(self.site_nbrRanges_idx).data.copy_(cpu_site_nbrRanges_idx)
            torch.flatten(self.site_nbr).data.copy_(cpu_site_nbr)
            torch.flatten(self.site_nbr_idx).data.copy_(cpu_site_nbr_idx)
            torch.flatten(self.site_nbrGroup_idx).data.copy_(cpu_site_nbrGroup_idx)
            torch.flatten(self.site_det_siteId).data.copy_(cpu_site_det_siteId)
            torch.flatten(self.site_curr_scl_siteId).data.copy_(cpu_site_curr_scl_siteId)
            torch.flatten(self.site_curr_scl_validIdx).data.copy_(cpu_site_curr_scl_validIdx)
            torch.flatten(self.site_curr_scl_idx).data.copy_(cpu_site_curr_scl_idx)
        else:
            lut_ff_legalization_cpp.initLegalize(
                pos, wlPrecond, torch.flatten(self.site_xy), self.pin_offset_x, self.pin_offset_y,
                sorted_net_idx, sorted_node_map, sorted_node_idx, self.flat_net2pin_map, 
                self.flat_net2pin_start_map, self.flop2ctrlSetId_map, self.flop_ctrlSets, 
                self.node2fence_region_map, self.node2outpinIdx_map, self.pin2net_map, self.pin2node_map,
                self.pin_typeIds, self.net2pincount, torch.flatten(self.site_types), 
                torch.flatten(self.spiral_accessor), self.site2addr_map, self.addr2site_map, 
                self.nodeNames, self.num_nets, self.num_nodes,
                self.num_sites_x, self.num_sites_y, self.num_clb_sites, self.num_threads, 
                self.SCL_IDX, self.nbrDistEnd, 
                self.nbrDistBeg, self.nbrDistIncr, self.numGroups, preClusteringMaxDist,
                self.WLscoreMaxNetDegree, self.maxList, spiralBegin, spiralEnd_maxD, self.net_bbox,
                self.net_pinIdArrayX, self.net_pinIdArrayY, self.flat_node2precluster_map,
                self.flat_node2prclstrCount, self.site_nbrRanges, self.site_nbrRanges_idx, self.site_nbrList,
                self.site_nbr, self.site_nbr_idx, self.site_nbrGroup_idx, self.site_det_siteId,
                self.site_curr_scl_siteId, self.site_curr_scl_validIdx, self.site_curr_scl_idx)

        #DBG
        #Preclustering Info
        preAll = (self.flat_node2prclstrCount[self.node2fence_region_map==0] > 1).sum().item()
        pre3 = (self.flat_node2prclstrCount[self.node2fence_region_map==0] > 2).sum().item()
        pre2 = preAll - pre3
        #print("# Precluster: ", preAll, " (", pre2, " + ", pre3, ")")
        #DBG
        print("Preclusters: %d (%d + %d) Initialization completed in %.3f seconds" % (preAll, pre2, pre3, time.time()-tt))
 
    # Greedy LG of macro LUT/FF Instances
    def greedyLG_macroInsts(self, pos, wlPrecond, node_z, sorted_node_map, sorted_node_idx, sorted_net_map, sorted_net_idx, sorted_pin_map):

        tt = time.time()
        spiralBegin = 0
        greedyExpansion = 5
        slotAssignFlowWeightScale = 1000.0
        slotAssignFlowWeightIncr = 0.5

        updXloc = torch.ones(self.num_nodes, dtype=pos.dtype, device=self.device)
        updXloc *= -1
        updYloc = torch.ones_like(updXloc)
        updYloc *= -1
        updZloc = torch.zeros(self.num_movable_nodes, dtype=torch.int, device=self.device)

        remInsts_mask = torch.logical_and(self.is_macro_inst == 1,self.node2fence_region_map < 2)
        num_remInsts = remInsts_mask.sum().item()
        rem_inst_areas = self.inst_areas[remInsts_mask]
        rem_inst_ids = torch.nonzero(remInsts_mask, as_tuple=True)[0].to(torch.int)

        #sorted node ids only comprise of remaining instances
        _, sorted_ids = torch.sort(rem_inst_areas, descending=True)
        sorted_remNode_idx = rem_inst_ids[sorted_ids]
        sorted_remNode_idx = sorted_remNode_idx.to(torch.int32)

        #sorted node map will consist of all instances sorted based on decreasing area
        _, sort_all_ids = torch.sort(self.inst_areas, descending=True)
        _, sorted_remNode_map = torch.sort(sort_all_ids)
        sorted_remNode_map = sorted_remNode_map.to(torch.int32)

        numFFs = self.node2fence_region_map[rem_inst_ids.long()].sum().item()
        numLUTs = rem_inst_ids.shape[0] - numFFs

        if pos.is_cuda:
            cpu_inst_curr_detSite = torch.flatten(self.inst_curr_detSite).cpu()
            cpu_site_det_sig_idx = torch.flatten(self.site_det_sig_idx).cpu()
            cpu_site_det_sig = torch.flatten(self.site_det_sig).cpu()
            cpu_site_det_impl_lut = torch.flatten(self.site_det_impl_lut).cpu()
            cpu_site_det_impl_ff = torch.flatten(self.site_det_impl_ff).cpu()
            cpu_site_det_impl_cksr = torch.flatten(self.site_det_impl_cksr).cpu()
            cpu_site_det_impl_ce = torch.flatten(self.site_det_impl_ce).cpu()
            cpu_site_det_siteId = torch.flatten(self.site_det_siteId).cpu()
            cpu_site_det_score = torch.flatten(self.site_det_score).cpu()
            cpu_node_x = updXloc.cpu()
            cpu_node_y = updYloc.cpu()
            cpu_node_z = updZloc.cpu()

            lut_ff_legalization_cpp.greedyLG_macroInsts(pos.cpu(), self.pin_offset_x.cpu(), self.pin_offset_y.cpu(), self.net_wts.cpu(),
                self.net_bbox.cpu(), wlPrecond.cpu(), torch.flatten(self.site_xy).cpu(), self.regionBox2xl.cpu(),
                self.regionBox2yl.cpu(), self.regionBox2xh.cpu(), self.regionBox2yh.cpu(), self.net_pinIdArrayX.cpu(),
                self.net_pinIdArrayY.cpu(), torch.flatten(self.spiral_accessor).cpu(),self.node2fence_region_map.cpu(),
                self.lut_type.cpu(), torch.flatten(self.site_types).cpu(), self.node2pincount.cpu(), self.net2pincount.cpu(),
                self.pin2net_map.cpu(), self.pin2node_map.cpu(), self.pin_typeIds.cpu(), self.flop2ctrlSetId_map.cpu(),
                self.flop_ctrlSets.cpu(), self.flat_net2pin_start_map.cpu(), self.flat_node2pin_start_map.cpu(),
                self.flat_node2pin_map.cpu(), self.flat_node2prclstrCount.cpu(), torch.flatten(self.flat_node2precluster_map).cpu(),
                self.node2regionBox_map.cpu(), sorted_net_map.cpu(), sorted_remNode_map.cpu(), sorted_remNode_idx.cpu(),
                self.node2outpinIdx_map.cpu(), self.flat_net2pin_map.cpu(),  self.addr2site_map.cpu(), self.site2addr_map.cpu(),
                self.nodeNames.cpu(), self.xl, self.yl, self.xh, self.yh, self.xWirelenWt, self.yWirelenWt, self.extNetCountWt,
                self.wirelenImprovWt, slotAssignFlowWeightScale, slotAssignFlowWeightIncr, self.NUM_BLE_PER_HALF_SLICE,
                num_remInsts, self.num_sites_x, self.num_sites_y, self.num_clb_sites, spiralBegin, self.spiral_accessor.shape[0],
                self.CKSR_IN_CLB, self.CE_IN_CLB, self.HALF_SLICE_CAPACITY, self.NUM_BLE_PER_SLICE, self.netShareScoreMaxNetDegree,
                self.WLscoreMaxNetDegree, greedyExpansion, self.SIG_IDX, self.num_threads, cpu_inst_curr_detSite, cpu_site_det_sig_idx,
                cpu_site_det_sig, cpu_site_det_impl_lut, cpu_site_det_impl_ff, cpu_site_det_impl_cksr, cpu_site_det_impl_ce,
                cpu_site_det_siteId, cpu_site_det_score, cpu_node_x, cpu_node_y, cpu_node_z)

            self.inst_curr_detSite.data.copy_(cpu_inst_curr_detSite.data)
            torch.flatten(self.site_det_sig_idx).data.copy_(cpu_site_det_sig_idx.data)
            torch.flatten(self.site_det_sig).data.copy_(cpu_site_det_sig.data)
            torch.flatten(self.site_det_impl_lut).data.copy_(cpu_site_det_impl_lut.data)
            torch.flatten(self.site_det_impl_ff).data.copy_(cpu_site_det_impl_ff.data)
            torch.flatten(self.site_det_impl_cksr).data.copy_(cpu_site_det_impl_cksr.data)
            torch.flatten(self.site_det_impl_ce).data.copy_(cpu_site_det_impl_ce.data)
            torch.flatten(self.site_det_siteId).data.copy_(cpu_site_det_siteId.data)
            torch.flatten(self.site_det_score).data.copy_(cpu_site_det_score.data)
            updXloc.data.copy_(cpu_node_x.data)
            updYloc.data.copy_(cpu_node_y.data)
            updZloc.data.copy_(cpu_node_z.data)
        else:
            lut_ff_legalization_cpp.greedyLG_macroInsts(pos, self.pin_offset_x, self.pin_offset_y, self.net_wts, self.net_bbox,
                wlPrecond, torch.flatten(self.site_xy), self.regionBox2xl, self.regionBox2yl, self.regionBox2xh, self.regionBox2yh,
                self.net_pinIdArrayX, self.net_pinIdArrayY, torch.flatten(self.spiral_accessor), self.node2fence_region_map,
                self.lut_type, torch.flatten(self.site_types), self.node2pincount, self.net2pincount, self.pin2net_map,
                self.pin2node_map, self.pin_typeIds, self.flop2ctrlSetId_map, self.flop_ctrlSets, self.flat_net2pin_start_map,
                self.flat_node2pin_start_map, self.flat_node2pin_map, self.flat_node2prclstrCount,
                torch.flatten(self.flat_node2precluster_map), self.node2regionBox_map, sorted_net_map, sorted_remNode_map,
                sorted_remNode_idx, self.node2outpinIdx_map, self.flat_net2pin_map, self.addr2site_map, self.site2addr_map, self.nodeNames,
                self.xl, self.yl, self.xh, self.yh, self.xWirelenWt, self.yWirelenWt, self.extNetCountWt, self.wirelenImprovWt,
                slotAssignFlowWeightScale, slotAssignFlowWeightIncr, self.NUM_BLE_PER_HALF_SLICE, num_remInsts, self.num_sites_x,
                self.num_sites_y, self.num_clb_sites, spiralBegin, self.spiral_accessor.shape[0], self.CKSR_IN_CLB, self.CE_IN_CLB,
                self.HALF_SLICE_CAPACITY, self.NUM_BLE_PER_SLICE, self.netShareScoreMaxNetDegree, self.WLscoreMaxNetDegree, greedyExpansion,
                self.SIG_IDX, self.num_threads, self.inst_curr_detSite, self.site_det_sig_idx, self.site_det_sig, self.site_det_impl_lut,
                self.site_det_impl_ff,  self.site_det_impl_cksr, self.site_det_impl_ce, self.site_det_siteId, self.site_det_score,
                updXloc, updYloc, updZloc)

        totalNodes = int(len(pos)/2)
        node_z.data.copy_(updZloc)
        mask = self.node2fence_region_map < 2
        pos[:self.num_nodes].data.masked_scatter_(mask, updXloc[mask])
        pos[totalNodes:totalNodes+self.num_nodes].data.masked_scatter_(mask, updYloc[mask])

        print("Greedy LG on %d macro insts (%d LUTs + %d FFs) takes %.3f seconds" % (num_remInsts, numLUTs, numFFs, time.time()-tt))
        return pos

    def runDLIter(self, pos, wlPrecond, sorted_node_map, sorted_node_idx, sorted_net_map, sorted_net_idx, sorted_pin_map, 
                  activeStatus, illegalStatus, dlIter):
        maxDist = 5.0
        spiralBegin = 0
        spiralEnd = 2 * (int(maxDist) + 1) * int(maxDist) + 1
        minStableIter = 3
        minNeighbors = 10
        cumsum_curr_scl = torch.zeros(self.num_clb_sites, dtype=torch.int32, device=self.device)
        sorted_clb_siteIds = torch.zeros_like(cumsum_curr_scl)
        validIndices_curr_scl = torch.ones_like(self.site_curr_scl_validIdx)
        validIndices_curr_scl *= -1

        if pos.is_cuda:
            lut_ff_legalization_cuda.runDLIter(pos, self.pin_offset_x, self.pin_offset_y, self.net_bbox, self.net_pinIdArrayX, 
                self.net_pinIdArrayY, torch.flatten(self.site_xy), torch.flatten(self.spiral_accessor), torch.flatten(self.site_types), 
                self.node2fence_region_map, self.lut_flop_indices, 
                self.flop_ctrlSets, self.flop2ctrlSetId_map, self.lut_type, self.flat_node2pin_start_map, self.flat_node2pin_map,
                self.node2pincount, self.net2pincount, self.pin2net_map, self.pin_typeIds, self.flat_net2pin_start_map, 
                self.pin2node_map, sorted_net_map, sorted_node_map, self.flat_node2prclstrCount, torch.flatten(self.flat_node2precluster_map),
                torch.flatten(self.site_nbrList), torch.flatten(self.site_nbrRanges), self.site_nbrRanges_idx, self.net_wts, self.addr2site_map, self.site2addr_map, self.num_sites_x, self.num_sites_y,
                self.num_clb_sites, self.num_lutflops, minStableIter, self.maxList, self.HALF_SLICE_CAPACITY, self.NUM_BLE_PER_SLICE, minNeighbors,
                spiralBegin, spiralEnd, self.int_min_val,
                self.numGroups, self.netShareScoreMaxNetDegree, self.WLscoreMaxNetDegree, maxDist, self.xWirelenWt, self.yWirelenWt, 
                self.wirelenImprovWt, self.extNetCountWt, self.CKSR_IN_CLB, self.CE_IN_CLB, self.SCL_IDX, self.SIG_IDX,
                self.site_nbr_idx, self.site_nbr, self.site_nbrGroup_idx, self.site_curr_pq_top_idx, self.site_curr_pq_sig_idx, 
                self.site_curr_pq_sig, self.site_curr_pq_idx, self.site_curr_stable, self.site_curr_pq_siteId, 
                self.site_curr_pq_validIdx, self.site_curr_pq_score, self.site_curr_pq_impl_lut, self.site_curr_pq_impl_ff, 
                self.site_curr_pq_impl_cksr, self.site_curr_pq_impl_ce, self.site_curr_scl_score, self.site_curr_scl_siteId,
                self.site_curr_scl_idx, cumsum_curr_scl, self.site_curr_scl_validIdx, validIndices_curr_scl, self.site_curr_scl_sig_idx, self.site_curr_scl_sig, 
                self.site_curr_scl_impl_lut, self.site_curr_scl_impl_ff, self.site_curr_scl_impl_cksr, self.site_curr_scl_impl_ce,
                self.site_next_pq_idx, self.site_next_pq_validIdx, self.site_next_pq_top_idx, self.site_next_pq_score, 
                self.site_next_pq_siteId, self.site_next_pq_sig_idx, self.site_next_pq_sig, self.site_next_pq_impl_lut, 
                self.site_next_pq_impl_ff, self.site_next_pq_impl_cksr, self.site_next_pq_impl_ce, self.site_next_scl_score,
                self.site_next_scl_siteId, self.site_next_scl_idx, self.site_next_scl_validIdx, self.site_next_scl_sig_idx, 
                self.site_next_scl_sig, self.site_next_scl_impl_lut, self.site_next_scl_impl_ff, self.site_next_scl_impl_cksr,
                self.site_next_scl_impl_ce, self.site_next_stable, self.site_det_score, self.site_det_siteId, self.site_det_sig_idx,
                self.site_det_sig, self.site_det_impl_lut, self.site_det_impl_ff, self.site_det_impl_cksr, self.site_det_impl_ce,
                self.inst_curr_detSite, self.inst_curr_bestScoreImprov, self.inst_curr_bestSite, self.inst_next_detSite,
                self.inst_next_bestScoreImprov, self.inst_next_bestSite, activeStatus, illegalStatus, self.inst_score_improv, self.site_score_improv, sorted_clb_siteIds)

        else:
            lut_ff_legalization_cpp.runDLIter(pos, self.pin_offset_x, self.pin_offset_y, self.net_bbox, self.net_pinIdArrayX, self.net_pinIdArrayY,
                torch.flatten(self.site_types), torch.flatten(self.site_xy), self.node2fence_region_map, self.flop_ctrlSets, self.flop2ctrlSetId_map, self.lut_type,
                self.flat_node2pin_start_map, self.flat_node2pin_map, self.node2pincount, self.net2pincount, self.pin2net_map, self.pin_typeIds, 
                self.flat_net2pin_start_map, self.pin2node_map, self.flat_node2prclstrCount, torch.flatten(self.flat_node2precluster_map), torch.flatten(self.site_nbrList),
                torch.flatten(self.site_nbrRanges), self.site_nbrRanges_idx, sorted_node_map, sorted_net_map, self.net_wts, self.addr2site_map,
                self.nodeNames, self.num_sites_x, self.num_sites_y, self.num_clb_sites, minStableIter, self.maxList, self.HALF_SLICE_CAPACITY, self.NUM_BLE_PER_SLICE,
                minNeighbors, self.numGroups, self.netShareScoreMaxNetDegree, self.WLscoreMaxNetDegree, self.xWirelenWt, self.yWirelenWt, self.wirelenImprovWt, self.extNetCountWt,
                self.CKSR_IN_CLB, self.CE_IN_CLB, self.SCL_IDX, self.PQ_IDX, self.SIG_IDX, self.num_nodes, self.num_threads, self.site_nbr_idx, self.site_nbr, self.site_nbrGroup_idx,
                self.site_curr_pq_top_idx, self.site_curr_pq_sig_idx, self.site_curr_pq_sig, self.site_curr_pq_idx, self.site_curr_pq_validIdx, self.site_curr_stable,
                self.site_curr_pq_siteId, self.site_curr_pq_score, self.site_curr_pq_impl_lut, self.site_curr_pq_impl_ff, self.site_curr_pq_impl_cksr, self.site_curr_pq_impl_ce,
                self.site_curr_scl_score, self.site_curr_scl_siteId, self.site_curr_scl_idx, self.site_curr_scl_validIdx, self.site_curr_scl_sig_idx, self.site_curr_scl_sig,
                self.site_curr_scl_impl_lut, self.site_curr_scl_impl_ff, self.site_curr_scl_impl_cksr, self.site_curr_scl_impl_ce, self.site_next_pq_idx, self.site_next_pq_validIdx,
                self.site_next_pq_top_idx, self.site_next_pq_score, self.site_next_pq_siteId, self.site_next_pq_sig_idx, self.site_next_pq_sig, self.site_next_pq_impl_lut,
                self.site_next_pq_impl_ff, self.site_next_pq_impl_cksr, self.site_next_pq_impl_ce, self.site_next_scl_score, self.site_next_scl_siteId, self.site_next_scl_idx,
                self.site_next_scl_validIdx, self.site_next_scl_sig_idx, self.site_next_scl_sig, self.site_next_scl_impl_lut, self.site_next_scl_impl_ff, self.site_next_scl_impl_cksr,
                self.site_next_scl_impl_ce, self.site_next_stable, self.site_det_score, self.site_det_siteId, self.site_det_sig_idx, self.site_det_sig, self.site_det_impl_lut,
                self.site_det_impl_ff, self.site_det_impl_cksr, self.site_det_impl_ce, self.inst_curr_detSite, self.inst_curr_bestScoreImprov, self.inst_curr_bestSite, self.inst_next_detSite,
                self.inst_next_bestScoreImprov, self.inst_next_bestSite, activeStatus, illegalStatus)

        ####DBG
        # print(dlIter,": ", (self.inst_curr_detSite[self.node2fence_region_map<2] > -1).sum().item(), "/", self.num_nodes)
        # print("\tactive Status : ", activeStatus.sum().item())
        # print("\tillegal Status : ", illegalStatus.sum().item())
        #DBG

    def ripUP_Greedy_slotAssign(self, pos, wlPrecond, node_z, sorted_node_map, sorted_node_idx, sorted_net_map, sorted_net_idx, sorted_pin_map):

        tt = time.time()
        spiralBegin = 0
        ripupExpansion = 1
        greedyExpansion = 5
        slotAssignFlowWeightScale = 1000.0
        slotAssignFlowWeightIncr = 0.5

        updXloc = torch.ones(self.num_nodes, dtype=pos.dtype, device=self.device)
        updXloc *= -1
        updYloc = torch.ones_like(updXloc)
        updYloc *= -1
        updZloc = torch.zeros(self.num_movable_nodes, dtype=torch.int, device=self.device)

        self.flat_node2precluster_map[:] = -1
        #Update first element as itself
        self.flat_node2precluster_map[:,0] = torch.arange(self.num_nodes, dtype=torch.int, device=self.device)
        self.flat_node2prclstrCount[:] = 1
        self.flat_node2prclstrCount[self.node2fence_region_map >1] = 0

        #RipUp + Greedy Legalization
        num_remInsts = (self.inst_curr_detSite == -1).sum().item()
        rem_inst_areas = self.inst_areas[self.inst_curr_detSite == -1]
        rem_inst_ids = torch.arange(self.num_nodes, device=self.device)[self.inst_curr_detSite ==  -1]

        #sorted node ids only comprise of remaining instances
        _, sorted_ids = torch.sort(rem_inst_areas, descending=True)
        sorted_remNode_idx = rem_inst_ids[sorted_ids]
        sorted_remNode_idx = sorted_remNode_idx.to(torch.int32)

        #sorted node map will consist of all instances sorted based on decreasing area
        _, sort_all_ids = torch.sort(self.inst_areas, descending=True)
        _, sorted_remNode_map = torch.sort(sort_all_ids)
        sorted_remNode_map = sorted_remNode_map.to(torch.int32)

        numFFs = self.node2fence_region_map[rem_inst_ids].sum().item()
        numLUTs = rem_inst_ids.shape[0] - numFFs
        #DBG
        #print("RipUP & Greedy LG on ", num_remInsts, " insts (", numLUTs, " LUTs + ", numFFs, " FFs)")
        #DBG

        if pos.is_cuda:
            cpu_inst_curr_detSite = torch.flatten(self.inst_curr_detSite).cpu()
            cpu_site_det_sig_idx = torch.flatten(self.site_det_sig_idx).cpu()
            cpu_site_det_sig = torch.flatten(self.site_det_sig).cpu()
            cpu_site_det_impl_lut = torch.flatten(self.site_det_impl_lut).cpu()
            cpu_site_det_impl_ff = torch.flatten(self.site_det_impl_ff).cpu()
            cpu_site_det_impl_cksr = torch.flatten(self.site_det_impl_cksr).cpu()
            cpu_site_det_impl_ce = torch.flatten(self.site_det_impl_ce).cpu()
            cpu_site_det_siteId = torch.flatten(self.site_det_siteId).cpu()
            cpu_site_det_score = torch.flatten(self.site_det_score).cpu()
            cpu_node_x = updXloc.cpu()
            cpu_node_y = updYloc.cpu()
            cpu_node_z = updZloc.cpu()

            lut_ff_legalization_cpp.ripUp_SlotAssign(pos.cpu(), self.pin_offset_x.cpu(), self.pin_offset_y.cpu(), self.net_wts.cpu(),
                self.net_bbox.cpu(), self.inst_areas.cpu(), wlPrecond.cpu(), torch.flatten(self.site_xy).cpu(),
                self.net_pinIdArrayX.cpu(), self.net_pinIdArrayY.cpu(), torch.flatten(self.spiral_accessor).cpu(),
                self.node2fence_region_map.cpu(), self.lut_type.cpu(), torch.flatten(self.site_types).cpu(), self.node2pincount.cpu(),
                self.net2pincount.cpu(), self.pin2net_map.cpu(), self.pin2node_map.cpu(), self.pin_typeIds.cpu(),
                self.flop2ctrlSetId_map.cpu(), self.flop_ctrlSets.cpu(), self.flat_node2pin_start_map.cpu(),
                self.flat_node2pin_map.cpu(), self.flat_net2pin_start_map.cpu(), self.flat_node2prclstrCount.cpu(), 
                torch.flatten(self.flat_node2precluster_map).cpu(), sorted_remNode_map.cpu(), sorted_remNode_idx.cpu(), sorted_net_map.cpu(),
                self.node2outpinIdx_map.cpu(), self.flat_net2pin_map.cpu(), 
                self.addr2site_map.cpu(), self.site2addr_map.cpu(), self.nodeNames.cpu(),
                self.nbrDistEnd, self.xWirelenWt, self.yWirelenWt, self.extNetCountWt, self.wirelenImprovWt, slotAssignFlowWeightScale,
                slotAssignFlowWeightIncr, self.NUM_BLE_PER_HALF_SLICE, num_remInsts, self.num_sites_x, self.num_sites_y, self.num_clb_sites, spiralBegin,
                self.spiral_accessor.shape[0], self.CKSR_IN_CLB, self.CE_IN_CLB, self.HALF_SLICE_CAPACITY, self.NUM_BLE_PER_SLICE,
                self.netShareScoreMaxNetDegree, self.WLscoreMaxNetDegree, ripupExpansion, greedyExpansion, self.SIG_IDX, self.num_threads,
                cpu_inst_curr_detSite, cpu_site_det_sig_idx, cpu_site_det_sig, cpu_site_det_impl_lut, cpu_site_det_impl_ff, cpu_site_det_impl_cksr,
                cpu_site_det_impl_ce, cpu_site_det_siteId, cpu_site_det_score, cpu_node_x, cpu_node_y, cpu_node_z)

            self.inst_curr_detSite.data.copy_(cpu_inst_curr_detSite.data)
            torch.flatten(self.site_det_sig_idx).data.copy_(cpu_site_det_sig_idx.data)
            torch.flatten(self.site_det_sig).data.copy_(cpu_site_det_sig.data)
            torch.flatten(self.site_det_impl_lut).data.copy_(cpu_site_det_impl_lut.data)
            torch.flatten(self.site_det_impl_ff).data.copy_(cpu_site_det_impl_ff.data)
            torch.flatten(self.site_det_impl_cksr).data.copy_(cpu_site_det_impl_cksr.data)
            torch.flatten(self.site_det_impl_ce).data.copy_(cpu_site_det_impl_ce.data)
            torch.flatten(self.site_det_siteId).data.copy_(cpu_site_det_siteId.data)
            torch.flatten(self.site_det_score).data.copy_(cpu_site_det_score.data)
            updXloc.data.copy_(cpu_node_x.data)
            updYloc.data.copy_(cpu_node_y.data)
            updZloc.data.copy_(cpu_node_z.data)
        else:
            lut_ff_legalization_cpp.ripUp_SlotAssign(pos, self.pin_offset_x, self.pin_offset_y, self.net_wts, self.net_bbox, self.inst_areas, wlPrecond,
                torch.flatten(self.site_xy), self.net_pinIdArrayX, self.net_pinIdArrayY, torch.flatten(self.spiral_accessor), 
                self.node2fence_region_map, self.lut_type, torch.flatten(self.site_types), self.node2pincount, self.net2pincount,
                self.pin2net_map, self.pin2node_map, self.pin_typeIds, self.flop2ctrlSetId_map, self.flop_ctrlSets, self.flat_node2pin_start_map,
                self.flat_node2pin_map, self.flat_net2pin_start_map, self.flat_node2prclstrCount, torch.flatten(self.flat_node2precluster_map),
                sorted_remNode_map, sorted_remNode_idx, sorted_net_map, self.node2outpinIdx_map, self.flat_net2pin_map, 
                self.addr2site_map, self.site2addr_map, self.nodeNames,
                self.nbrDistEnd, self.xWirelenWt, self.yWirelenWt, self.extNetCountWt, self.wirelenImprovWt, slotAssignFlowWeightScale,
                slotAssignFlowWeightIncr, self.NUM_BLE_PER_HALF_SLICE, num_remInsts, self.num_sites_x, self.num_sites_y, self.num_clb_sites, spiralBegin,
                self.spiral_accessor.shape[0], self.CKSR_IN_CLB, self.CE_IN_CLB, self.HALF_SLICE_CAPACITY, self.NUM_BLE_PER_SLICE,
                self.netShareScoreMaxNetDegree, self.WLscoreMaxNetDegree, ripupExpansion, greedyExpansion, self.SIG_IDX, self.num_threads,
                self.inst_curr_detSite, self.site_det_sig_idx, self.site_det_sig, self.site_det_impl_lut, self.site_det_impl_ff, 
                self.site_det_impl_cksr, self.site_det_impl_ce, self.site_det_siteId, self.site_det_score, updXloc, updYloc, updZloc)

        totalNodes = int(len(pos)/2)
        node_z.data.copy_(updZloc)
        mask = self.node2fence_region_map < 2
        pos[:self.num_nodes].data.masked_scatter_(mask, updXloc[mask])
        pos[totalNodes:totalNodes+self.num_nodes].data.masked_scatter_(mask, updYloc[mask])

        print("RipUP & Greedy LG on %d insts (%d LUTs + %d FFs) takes %.3f seconds" % (num_remInsts, numLUTs, numFFs, time.time()-tt))
        return pos
