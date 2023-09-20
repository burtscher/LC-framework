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

#ifndef in_2d
#define in_2d(a, b) in[(b) * x + (a)]
#endif


#ifndef out_2d
#define out_2d(a, b) out[(b) * x + (a)]
#endif

template <typename T>
static __global__ void d_lorenzo_encode_2d(const T* const in, T* const out, const int x)
{
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) out_2d(0, 0) = in_2d(0, 0);
    for (int i = threadIdx.x + 1; i < x; i += TPB) {
      out_2d(i, 0) = in_2d(i, 0) - in_2d(i - 1, 0);
    }
  } else {
    const int j = blockIdx.x;
    if (threadIdx.x == 0) out_2d(0, j) = in_2d(0, j) - in_2d(0, j - 1);
    for (int i = threadIdx.x + 1; i < x; i += TPB) {
      out_2d(i, j) = in_2d(i, j) - in_2d(i - 1, j) - in_2d(i, j - 1) + in_2d(i - 1, j - 1);
    }
  }
}


template <typename T>  // Use same encoder structure, one block per row, take care of that row with that block
static __global__ void d_lorenzo_decode_2d_x(const T* const in, T* const out, const int x)
{
  // Each block gets one X 'slice' to prefix sum, need Y blocks
  const int tid = threadIdx.x;
  const int beg = tid * x / TPB;
  const int end = (tid + 1) * x / TPB;
  const int j = blockIdx.x;
  __shared__ T temp [32];

  // compute local sum
  T sum = 0;
  for (int i = beg; i < end; i++) {
    sum += in_2d(i, j);
  }

  // block-wise inclusive prefix sum
  sum = block_prefix_sum(sum, temp);

  // compute intermediate values
  for (int i = end - 1; i >= beg; i--) {
    out_2d(i, j) = sum;
    sum -= in_2d(i, j);
  }
}


template <typename T>
static __global__ void d_lorenzo_decode_2d_y(const T* const in, T* const out, const int y)
{
  // Each block gets one Y 'slice' to prefix sum, need X blocks
  const int x = gridDim.x;
  const int tid = threadIdx.x;
  const int beg = tid * y / TPB;
  const int end = (tid + 1) * y / TPB;
  const int i = blockIdx.x;
  __shared__ T temp [32];

  // compute local sum
  T sum = 0;
  for (int j = beg; j < end; j++) {
    sum += in_2d(i, j);
  }

  // block-wise inclusive prefix sum
  sum = block_prefix_sum(sum, temp);

  // compute intermediate values
  for (int j = end - 1; j >= beg; j--) {
    out_2d(i, j) = sum;
    sum -= in_2d(i, j);
  }
}


static inline void d_LOR2D_i32(int& size, byte*& data, const int paramc, const double paramv [])
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

  // Get params
  if (paramc != 2) {
    fprintf(stderr, "ERROR: Lorenzo 2D needs an x and y as parameters\n");
    exit(-1);
  }
  const int x = (int)paramv[0];
  const int y = (int)paramv[1];
  if (x * y != insize) {
    fprintf(stderr, "ERROR: X(%d) and Y(%d) don't match size %d\n", x, y, insize);
    exit(-1);
  }

  // Encode
  d_lorenzo_encode_2d<<<y, TPB>>>(in_t, d_encoded, x);
  cudaDeviceSynchronize();

  // Finalize
  data = (byte*)d_encoded;
  cudaFree(in_t);
  return;
}


static inline void d_iLOR2D_i32(int& size, byte*& data, const int paramc, const double paramv [])
{
  using type = int;
  type* in_t = (type*)data;
  // Check that size is correct
  if (size % sizeof(type) != 0) {
    fprintf(stderr, "ERROR: size %d is not evenly divisible by type size %ld\n", size, sizeof(type));
    exit(-1);
  }
  type* d_decoded;
  if (cudaSuccess != cudaMalloc((void **)&d_decoded, size)) fprintf(stderr, "CUDA ERROR: could not allocate d_encoded\n");
  int insize = size / sizeof(type);

  // Get params
  if (paramc != 2) {
    fprintf(stderr, "ERROR: Lorenzo 2D needs an x and y as parameters\n");
    exit(-1);
  }
  const int x = (int)paramv[0];
  const int y = (int)paramv[1];
  if (x * y != insize) {
    fprintf(stderr, "ERROR: X(%d) and Y(%d) don't match size %d\n", x, y, insize);
    exit(-1);
  }

  // Decode
  d_lorenzo_decode_2d_x<<<y, TPB>>>(in_t, d_decoded, x);
  d_lorenzo_decode_2d_y<<<x, TPB>>>(d_decoded, in_t, y); // The encoded/decoded swap is intentional here
  cudaDeviceSynchronize();

  // Finalize
  cudaFree(d_decoded);
  return;
}