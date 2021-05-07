############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 2015 Xilinx Inc. All rights reserved.
############################################################
open_project -reset myproject_prj
set_top myproject
add_files firmware/myproject.cpp -cflags "-I[file normalize ../../nnet_utils]"
add_files -tb myproject_test.cpp -cflags "-I[file normalize ../../nnet_utils]"
add_files -tb firmware/weights
#add_files -tb tb_data
open_solution -reset "solution1"
set_part {xc7vx690tffg1927-2}
#set_part {xcvu9p-flgb2104-2-i}
create_clock -period 5 -name default
#source "./fir_hls_prj/solution1/directives.tcl"
csim_design
csynth_design
#cosim_design -trace_level all
#export_design -format ip_catalog
exit
