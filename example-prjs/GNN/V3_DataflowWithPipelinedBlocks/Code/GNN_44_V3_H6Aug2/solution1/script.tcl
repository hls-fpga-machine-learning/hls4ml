############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project GNN_44_V3_H6Aug2R1
set_top myproject
add_files GNN_44_V3_H6Aug2R1/myproject.cpp
add_files GNN_44_V3_H6Aug2R1/myproject.h
add_files GNN_44_V3_H6Aug2R1/nnet_activation.h
add_files GNN_44_V3_H6Aug2R1/nnet_batchnorm.h
add_files GNN_44_V3_H6Aug2R1/nnet_common.h
add_files GNN_44_V3_H6Aug2R1/nnet_conv.h
add_files GNN_44_V3_H6Aug2R1/nnet_conv2d.h
add_files GNN_44_V3_H6Aug2R1/nnet_dense.h
add_files GNN_44_V3_H6Aug2R1/nnet_dense_large.h
add_files GNN_44_V3_H6Aug2R1/nnet_graph.h
add_files GNN_44_V3_H6Aug2R1/nnet_helpers.h
add_files GNN_44_V3_H6Aug2R1/nnet_merge.h
add_files GNN_44_V3_H6Aug2R1/nnet_pooling.h
add_files GNN_44_V3_H6Aug2R1/parameters.h
add_files -tb GNN_44_V3_H6Aug2R1/myproject_test.cpp
open_solution "solution1"
set_part {xcku115-flva1517-1-c} -tool vivado
create_clock -period 10 -name default
#source "./GNN_44_V3_H6Aug2R1/solution1/directives.tcl"
csim_design -clean
csynth_design
cosim_design -rtl vhdl
export_design -rtl vhdl -format ip_catalog
