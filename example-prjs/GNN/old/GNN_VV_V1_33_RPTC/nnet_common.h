//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#include "ap_fixed.h"

namespace nnet {

// Common type definitions
enum io_type {io_parallel = 0, io_serial};

// Default data types (??) TODO: Deprecate
typedef ap_fixed<16,4>  weight_t_def;
typedef ap_fixed<16,4>  bias_t_def;
typedef ap_fixed<32,10> accum_t_def;

 template<class data_T, int NIN1, int NIN2>
   void merge(
	      data_T data1[NIN1], 
	      data_T data2[NIN2],
	      data_T res[NIN1+NIN2])
 {
   for(int ii=0; ii<NIN1; ii++){
     res[ii] = data1[ii];
   }
   for(int ii=0; ii<NIN2; ii++){
     res[NIN1+ii] = data2[ii];
   }
 }

 template<class data_T, int NSHARED, int NIN1, int NIN2>
   void merge2d(
		data_T data1[NSHARED][NIN1], 
		data_T data2[NSHARED][NIN2],
		data_T res[NSHARED][NIN1+NIN2])
 {
   for(int ii=0; ii<NSHARED; ii++){
     for(int jj=0; jj<NIN1; jj++){
       res[ii][jj] = data1[ii][jj];
     }
     for(int jj=0; jj<NIN2; jj++){
       res[ii][NIN1+jj] = data2[ii][jj];
     }
   }
 }
 
}

#endif
