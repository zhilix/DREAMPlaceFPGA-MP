##
# @file   macro_graph.py
# @author Zhili Xiong(MLCAD contest 2023)
# @date   Jul 2021
#

import sys
import os
import re
import math
import time 
import numpy as np 
import logging
import Params
import torch
import pdb
from torch_geometric.data import Data
from collections import deque
import networkx as nx


class MacroGraph(object):
    """
    @brief MacroGraph class
    """
    def __init__(self, num_nodes, num_movable_nodes, macro_mask, num_nets, 
                 net_weights, net_mask, flat_net2pin, flat_net2pin_start, 
                 pin2node_map, pin_types, cascade_inst_names, flat_cascade_inst2node_start, 
                 flat_cascade_inst2node, node_size_x, node_size_y, flat_node2pin_start_map, node_types):
        """
        @brief initialization
        """
        self.num_nodes = num_nodes
        self.num_movable_nodes = num_movable_nodes
        self.macro_mask = macro_mask
        self.num_nets = num_nets
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.flat_net2pin = flat_net2pin
        self.net2pin_start = flat_net2pin_start
        self.pin2node_map = pin2node_map
        self.pin_types = pin_types

        self.cascade_inst_names = cascade_inst_names
        self.flat_cascade_inst2node_start = flat_cascade_inst2node_start
        self.flat_cascade_inst2node = flat_cascade_inst2node

        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.node_types = node_types

        # write out hypergraph file for hypergraph partitioning using hMETIS binary
        self.hGraph_file = "design.hgr"
        self.WriteGraphFile(self.hGraph_file)

        for node_id in range(self.num_nodes):
            if self.macro_mask[node_id]==1 and self.node_types[node_id] not in ['RAMB36E2', 'DSP48E2']:
                self.macro_mask[node_id]=0
                print('Node_id: ', node_id, ' is changed to nonMarco: ', self.node_types[node_id])

        # set of parameters for hMETIS
        self.num_macro_inst = sum(self.macro_mask)
        # the number of part should be thoushands
        self.num_parts = int((self.num_movable_nodes - self.num_macro_inst) / 150)
        # the unbalance factor should be in the range of 1-49
        self.uB_factor = 30

        # USAGE: shmetis HGraphFile FixFile Nparts UBfactor 
        # or shmetis HGraphFile Nparts UBfactor
        cmd = "./thirdparty/shmetis %s %s %s" % (self.hGraph_file, self.num_parts, self.uB_factor)
        os.system(cmd)

        # read partition file from hMETIS
        # partition non-macro and macro nodes together
        partition = self.ReadPartitionFile("design.hgr.part.%d" % (self.num_parts))

        self.node2cluster_map, self.num_clusters = self.cluster_nodes(partition)

        self.cluster_adj_matrix, self.cluster_adj_dir_matrix = self.build_cluster_adj_matrix()

        levels, order = self.compute_node_levels()

        self.num_features = 2 # is_macro, area
        self.data = self.build_geometric_data()

    
    def WriteGraphFile(self, hGraph_file):
        """
        @brief write out graph file for hypergraph partitioning using hMETIS
        """
        tt = time.time()
        logging.info("writing to %s" % (hGraph_file))

        content = ""

        # write out the hypergraph summary
        # 1st line: first integer is the number of hyperedges (|Eh|), the second is the number of vertices (|V|)
        num_hyperedges = sum(self.net_mask)
        num_vertices = self.num_nodes
        content += "%d %d\n" % (num_hyperedges, num_vertices)

        # write out the hyperedges
        for j in range(self.num_nets):
            if self.net_mask[j] == 1:
                for i in range(self.net2pin_start[j], self.net2pin_start[j+1]):
                    pin_id = self.flat_net2pin[i]
                    node_id = self.pin2node_map[pin_id]
                    content += "%d " % (node_id)

                content += "\n"

        with open(hGraph_file, 'w') as f:
            f.write(content)

    
    def ReadPartitionFile(self, partition_file):
        """
        @brief read partition file from hMETIS
        """
        tt = time.time()
        logging.info("reading from %s" % (partition_file))

        # read partition file
        partition = np.zeros(self.num_nodes, dtype=np.int32)
        with open(partition_file, 'r') as f:
            node_id = 0
            for line in f:
                partition[node_id] = int(line)
                node_id += 1

        logging.info("Reading partition file takes %.2f seconds" % (time.time()-tt))

        return partition

    def cluster_nodes(self, partition):
        """
        @brief cluster nodes into groups
        """
        tt = time.time()
        # cluster map
        node2cluster_map = np.zeros(self.num_nodes, dtype=np.int32)
        cascade_mask = np.zeros(self.num_nodes, dtype=np.int32)

        # First cluster all the cascaded macros
        num_cascade_inst = len(self.cascade_inst_names)
        for i in range(num_cascade_inst):
            for j in range(self.flat_cascade_inst2node_start[i], self.flat_cascade_inst2node_start[i+1]):
                node_id = self.flat_cascade_inst2node[j]
                cascade_mask[node_id] = 1
                node2cluster_map[node_id] = i + self.num_parts

        # make sure all the cascaded macros are masked 
        self.macro_mask = self.macro_mask | cascade_mask
            
        # Cluster all the non-cascaded macros and non-macro nodes
        non_cascade_macro_cnt = 0
        for i in range(self.num_nodes):
            # Then cluster all the non-cascaded macros
            if self.macro_mask[i] == 1 and cascade_mask[i] == 0:
                node2cluster_map[i] = non_cascade_macro_cnt + self.num_parts + num_cascade_inst
                non_cascade_macro_cnt += 1

            # Finally cluster all the non-macro nodes
            elif self.macro_mask[i] == 0:
                node2cluster_map[i] = partition[i]
        
        num_clusters = self.num_parts + num_cascade_inst + non_cascade_macro_cnt

        logging.info("Clustering takes %.2f seconds" % (time.time()-tt))

        return node2cluster_map, num_clusters
    
    def build_cluster_adj_matrix(self):
        """
        @brief build cluster adjacency matrix
        """
        tt = time.time()
        # build cluster adjacency matrix
        # cluster_adj_matrix = torch.tensor(self.num_clusters, self.num_clusters, dtype=torch.long)
        cluster_adj_matrix = np.zeros((self.num_clusters, self.num_clusters), dtype=np.int32)
        cluster_adj_dir_matrix = np.zeros((self.num_clusters, self.num_clusters), dtype=np.int32)
        
        for i in range(self.num_nets):
            if self.net_mask[i] == 1:
                src_pins = []
                dst_pins = []
                for j in range(self.net2pin_start[i], self.net2pin_start[i+1]):
                    pin_id = self.flat_net2pin[j]
                    
                    # pin_type = 0: src pin_type = 1: snk
                    if self.pin_types[pin_id] == 0:
                        src_pins.append(pin_id)
                    else:
                        dst_pins.append(pin_id)

                for src_pin in src_pins:
                    for dst_pin in dst_pins:
                        src_node_id = self.pin2node_map[src_pin]
                        dst_node_id = self.pin2node_map[dst_pin]

                        src_cluster_id = self.node2cluster_map[src_node_id]
                        dst_cluster_id = self.node2cluster_map[dst_node_id]

                        if src_cluster_id != dst_cluster_id:
                            cluster_adj_matrix[src_cluster_id][dst_cluster_id] = 1
                            cluster_adj_matrix[dst_cluster_id][src_cluster_id] = 1
                            cluster_adj_dir_matrix[src_cluster_id][dst_cluster_id] = 1

        logging.info("Building cluster adjacency matrix takes %.2f seconds" % (time.time()-tt))

        return cluster_adj_matrix,cluster_adj_dir_matrix

    def build_geometric_data(self):
        """
        @brief build geometric data for graph neural network
        """

        # build cluster feature matrix
        cluster_feature_matrix = torch.zeros(self.num_clusters, 2, dtype=torch.float)

        # build cluster edge matrix
        self.cluster_adj_matrix = torch.tensor(self.cluster_adj_matrix, dtype=torch.long)
        cluster_edge_index = self.cluster_adj_matrix.nonzero().t().contiguous()

        data = Data(x=cluster_feature_matrix, edge_index=cluster_edge_index)

        # pdb.set_trace()

        return data
    
    def compute_node_levels(self):
        num_clusters = self.num_clusters
        num_marcos = self.num_clusters - self.num_parts
        levels = [None] * num_clusters
        processed = [False] * num_clusters
        order = []
        Marco_num_processed = 0
        marco_queue = []
        stdcell_queue = []

        while Marco_num_processed < num_marcos:  # Continue until all nodes are processed
            # Find the first unprocessed Marco node
            src_node = next((i for i in range(self.num_parts, num_clusters) if not processed[i]), None)
            if src_node is None:  # No unprocessed Marco nodes left
                break

            # Forward propagation
            levels[src_node] = 1
            marco_queue.append(src_node)
            processed[src_node] = True
            Marco_num_processed += 1

            while marco_queue or stdcell_queue:
                if Marco_num_processed == num_marcos:
                    break
                
                current_node = None
                if marco_queue:
                    current_node = marco_queue.pop(0)
                elif stdcell_queue:
                    current_node = stdcell_queue.pop(0)

                for dst_node in range(num_clusters):
                    if self.cluster_adj_matrix[current_node][dst_node] and not processed[dst_node]:
                        levels[dst_node] = levels[current_node] + 1
                        if dst_node < self.num_parts:  # Standard cell node
                            stdcell_queue.append(dst_node)
                        else:  # Marco node
                            marco_queue.append(dst_node)
                            Marco_num_processed += 1

                        processed[dst_node] = True


            # Backward propagation
            for src_node in range(self.num_parts, num_clusters):
                if Marco_num_processed == num_marcos:
                    break
                if not processed[src_node]:
                    min_level = float('inf')
                    for dst_node in range(num_clusters):
                        if self.cluster_adj_matrix[src_node][dst_node] and processed[dst_node]:
                            min_level = min(min_level, levels[dst_node])

                    levels[src_node] = min_level - 1 if min_level != float('inf') else 1
                    processed[src_node] = True
                    Marco_num_processed += 1

        self.area, self.pin_num, self.cluster_type = self.get_ranking_data()
        Marco_pin_num = self.pin_num[self.num_parts:]
        Marco_area = self.area[self.num_parts:]
        Marco_level = levels[self.num_parts:]


        def sort_key(node_id):
            num_pins = Marco_pin_num[node_id]
            size = Marco_area[node_id]
            level = Marco_level[node_id]
            return (-level, -num_pins, -size, node_id)

        order = sorted(range(self.num_clusters - self.num_parts), key=sort_key) 
        order = [value + self.num_parts for value in order] #return the cluster id, in placing order

        return levels, order
    
    def get_ranking_data(self):
        area = np.zeros((self.num_clusters))
        pin_num = np.zeros((self.num_clusters))
        cluster_type = ['std_node']*self.num_clusters

        for node_id, isMarco in enumerate(self.macro_mask):
            cluster_id = self.node2cluster_map[node_id]
            area[cluster_id] += self.node_size_x[node_id]*self.node_size_y[node_id]
            pin_num[cluster_id] += self.flat_node2pin_start_map[node_id+1] - self.flat_node2pin_start_map[node_id]
            if isMarco==1 and cluster_type[cluster_id] is 'std_node':
                cluster_type[cluster_id] = self.node_types[node_id]
        return area, pin_num, cluster_type