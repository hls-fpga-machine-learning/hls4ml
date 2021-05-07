############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project GNN_33_V3_H6Aug2
set_top myproject
add_files GNN_33_V3_H6Aug2/myproject.cpp
add_files GNN_33_V3_H6Aug2/myproject.h
add_files GNN_33_V3_H6Aug2/nnet_activation.h
add_files GNN_33_V3_H6Aug2/nnet_batchnorm.h
add_files GNN_33_V3_H6Aug2/nnet_common.h
add_files GNN_33_V3_H6Aug2/nnet_conv.h
add_files GNN_33_V3_H6Aug2/nnet_conv2d.h
add_files GNN_33_V3_H6Aug2/nnet_dense.h
add_files GNN_33_V3_H6Aug2/nnet_dense_large.h
add_files GNN_33_V3_H6Aug2/nnet_graph.h
add_files GNN_33_V3_H6Aug2/nnet_helpers.h
add_files GNN_33_V3_H6Aug2/nnet_merge.h
add_files GNN_33_V3_H6Aug2/nnet_pooling.h
add_files GNN_33_V3_H6Aug2/parameters.h
add_files -tb GNN_33_V3_H6Aug2/myproject_test.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xcku115-flva1517-1-c} -tool vivado
create_clock -period 10 -name default
#source "./GNN_33_V3_H6Aug2/solution1/directives.tcl"
csim_design -clean
csynth_design
cosim_design -rtl vhdl
export_design -format ip_catalog
