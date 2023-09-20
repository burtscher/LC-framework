/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2023, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cub/cub.cuh>


template <typename T>
static __global__ void d_lorenzo_encode_1d(const T* const in, T* const out, const int size_in_T)
{
  const int tid = threadIdx.x + blockIdx.x * TPB;
  __shared__ T shared_buf [TPB + 1];
  if (threadIdx.x == 0) shared_buf[0] = (tid == 0) ? 0 : in[tid - 1];
  if (tid < size_in_T) {
    shared_buf[threadIdx.x + 1] = in[tid];
  }
  __syncthreads();
  if (tid < size_in_T) {
    out[tid] = shared_buf[threadIdx.x + 1] - shared_buf[threadIdx.x];
  }
}


static inline void d_LOR1D_i32(int& size, byte*& data, const int paramc, const double paramv [])
{
  using type = int;
  type* in_t = (type*)data;
  // Check that size is correct
  if (size % sizeof(type) != 0) {
    fprintf(stderr, "ERROR: size %d is not evenly divisible by type size %ld\n", size, sizeof(type));
    exit(-1);
  }
  type* d_encoded;
  if (cudaSuccess != cudaMalloc((void **)&d_encoded, size)) fprintf(stderr, "CUDA ERROR: could not allocate d_encoded\n");
  int insize = size / sizeof(type);

  // Encode
  const int blocks = (insize + TPB - 1) / TPB;
  d_lorenzo_encode_1d<<<blocks, TPB>>>(in_t, d_encoded, insize);
  cudaDeviceSynchronize();

  // Finalize
  data = (byte*)d_encoded;
  cudaFree(in_t);
  return;
}


static inline void d_iLOR1D_i32(int& size, byte*& data, const int paramc, const double paramv [])
{
  using type = int;
  type* in_t = (type*)data;
  // Check that size is correct
  if (size % sizeof(type) != 0) {
    fprintf(stderr, "ERROR: size %d is not evenly divisible by type size %ld\n", size, sizeof(type));
    exit(-1);
  }
  type* d_decoded;
  if (cudaSuccess != cudaMalloc((void **)&d_decoded, size)) fprintf(stderr, "CUDA ERROR: could not allocate d_decoded\n");
  int insize = size / sizeof(type);

  // Determine temporary device storage requirements for inclusive prefix sum
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in_t, d_decoded, insize);
  if (cudaSuccess != cudaMalloc(&d_temp_storage, temp_storage_bytes)) fprintf(stderr, "CUDA ERROR: could not allocate d_temp_storage\n");

  // Launch CUB scan (decode)
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in_t, d_decoded, insize);
  cudaDeviceSynchronize();

  // Finalize
  data = (byte*)d_decoded;
  cudaFree(in_t);
  return;
}