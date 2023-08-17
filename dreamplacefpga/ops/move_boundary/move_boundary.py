##
# @file   move_boundary.py
# @author Rachel Selina (DREAMPlaceFPGA), Yibo Lin (DREAMPlace)
# @date   Aug 2023
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import dreamplacefpga.ops.move_boundary.move_boundary_cpp as move_boundary_cpp
import dreamplacefpga.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplacefpga.ops.move_boundary.move_boundary_cuda as move_boundary_cuda

import pdb

class MoveBoundaryFunction(Function):
    """ 
    @brief Bound cells into layout boundary, perform in-place update 
    """
    @staticmethod
    def forward(
          pos,
          node_size_x,
          node_size_y,
          regionBox2xl,
          regionBox2yl,
          regionBox2xh,
          regionBox2yh,
          node2regionBox_map,
          xl, 
          yl, 
          xh, 
          yh, 
          num_movable_nodes, 
          num_filler_nodes, 
          num_threads
          ):

        if pos.is_cuda:
            output = move_boundary_cuda.forward(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    regionBox2xl,
                    regionBox2yl,
                    regionBox2xh,
                    regionBox2yh,
                    node2regionBox_map,
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    num_movable_nodes, 
                    num_filler_nodes
                    )
        else:
            output = move_boundary_cpp.forward(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    regionBox2xl,
                    regionBox2yl,
                    regionBox2xh,
                    regionBox2yh,
                    node2regionBox_map,
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    num_movable_nodes, 
                    num_filler_nodes, 
                    num_threads
                    )
        return output

class MoveBoundary(object):
    """ 
    @brief Bound cells into layout boundary, perform in-place update 
    """
    def __init__(self, placedb, node_size_x, node_size_y,
                 regionBox2xl, regionBox2yl, regionBox2xh, regionBox2yh,
                 node2regionBox_map, num_threads):
        super(MoveBoundary, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.regionBox2xl = regionBox2xl
        self.regionBox2yl = regionBox2yl
        self.regionBox2xh = regionBox2xh
        self.regionBox2yh = regionBox2yh
        self.node2regionBox_map = node2regionBox_map
        self.xl = placedb.xl 
        self.yl = placedb.yl
        self.xh = placedb.xh 
        self.yh = placedb.yh 
        self.num_movable_nodes = placedb.num_movable_nodes
        self.num_filler_nodes = placedb.num_filler_nodes
        self.num_threads = num_threads
    def __call__(self, pos): 
        return MoveBoundaryFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                regionBox2xl = self.regionBox2xl,
                regionBox2yl = self.regionBox2yl,
                regionBox2xh = self.regionBox2xh,
                regionBox2yh = self.regionBox2yh,
                node2regionBox_map = self.node2regionBox_map,
                xl=self.xl, 
                yl=self.yl, 
                xh=self.xh, 
                yh=self.yh, 
                num_movable_nodes=self.num_movable_nodes, 
                num_filler_nodes=self.num_filler_nodes, 
                num_threads=self.num_threads
                )
