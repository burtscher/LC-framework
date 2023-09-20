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

#ifndef in_3d
#define in_3d(a, b, c) in[((c) * y + (b)) * x + (a)]
#endif


#ifndef out_3d
#define out_3d(a, b, c) out[((c) * y + (b)) * x + (a)]
#endif


template <typename T>
static __global__ void d_lorenzo_encode_3d_corner_edges_surfaces(const T* const in, T* const out, const int x, const int y, const int z)
{
  if (blockIdx.x == 0) {
    // Compute corner element
    if (threadIdx.x == 0) out_3d(0, 0, 0) = in_3d(0, 0, 0);

    // Compute edge elements
    for (int i = threadIdx.x + 1; i < x; i += TPB) {
      out_3d(i, 0, 0) = in_3d(i, 0, 0) - in_3d(i - 1, 0, 0);
    }
    for (int j = threadIdx.x + 1; j < y; j += TPB) {
      out_3d(0, j, 0) = in_3d(0, j, 0) - in_3d(0, j - 1, 0);
    }
    for (int k = threadIdx.x + 1; k < z; k += TPB) {
      out_3d(0, 0, k) = in_3d(0, 0, k) - in_3d(0, 0, k - 1);
    }
  } else {
    // Compute surface elements
    const int b = blockIdx.x;
    if (b < y) {
      const int j = b;
      for (int i = threadIdx.x + 1; i < x; i += TPB) {
        out_3d(i, j, 0) = in_3d(i, j, 0) - in_3d(i - 1, j, 0) - in_3d(i, j - 1, 0) + in_3d(i - 1, j - 1, 0);
      }
    }
    if (b < z) {
      const int k = b;
      for (int i = threadIdx.x + 1; i < x; i += TPB) {
        out_3d(i, 0, k) = in_3d(i, 0, k) - in_3d(i - 1, 0, k) - in_3d(i, 0, k - 1) + in_3d(i - 1, 0, k - 1);
      }
      for (int j = threadIdx.x + 1; j < y; j += TPB) {
        out_3d(0, j, k) = in_3d(0, j, k) - in_3d(0, j - 1, k) - in_3d(0, j, k - 1) + in_3d(0, j - 1, k - 1);
      }
    }
  }
}


template <typename T>
static __global__ void d_lorenzo_encode_3d_interior(const T* const in, T* const out, const int x, const int y)
{
  const int k = blockIdx.x + 1;
  const int j = blockIdx.y + 1;
  for (int i = threadIdx.x + 1; i < x; i += TPB) {
    out_3d(i, j, k) = in_3d(i, j, k) - in_3d(i - 1, j, k) - in_3d(i, j - 1, k) - in_3d(i, j, k - 1) + in_3d(i - 1, j - 1, k) + in_3d(i - 1, j, k - 1) + in_3d(i, j - 1, k - 1) - in_3d(i - 1, j - 1, k - 1);
  }
}


template <typename T>
static __global__ void d_lorenzo_decode_3d_x(const T* const in, T* const out, const int x, const int y)
{
  // Each block gets one X 'slice' to prefix sum, need Y * Z blocks
  const int tid = threadIdx.x;
  const int beg = tid * x / TPB;
  const int end = (tid + 1) * x / TPB;
  const int j = blockIdx.x;
  const int k = blockIdx.y;
  __shared__ T temp [32];

  // compute local sum
  T sum = 0;
  for (int i = beg; i < end; i++) {
    sum += in_3d(i, j, k);
  }

  // block-wise inclusive prefix sum
  sum = block_prefix_sum(sum, temp);

  // compute intermediate values
  for (int i = end - 1; i >= beg; i--) {
    out_3d(i, j, k) = sum;
    sum -= in_3d(i, j, k);
  }
}


template <typename T>
static __global__ void d_lorenzo_decode_3d_y(T* const out, const int x, const int y)
{
  // Each block gets one Y 'slice' to prefix sum, need X * Z blocks
  const int tid = threadIdx.x;
  const int beg = tid * y / TPB;
  const int end = (tid + 1) * y / TPB;
  const int i = blockIdx.x;
  const int k = blockIdx.y;
  __shared__ T temp [32];

  // compute local sum
  T sum = 0;
  for (int j = beg; j < end; j++) {
    sum += out_3d(i, j, k);
  }

  // block-wise inclusive prefix sum
  sum = block_prefix_sum(sum, temp);

  // compute intermediate values
  for (int j = end - 1; j >= beg; j--) {
    const T val = out_3d(i, j, k);
    out_3d(i, j, k) = sum;
    sum -= val;
  }
}


template <typename T>
static __global__ void d_lorenzo_decode_3d_z(T* const out, const int x, const int y, const int z)
{
  // Each block gets one Z 'slice' to prefix sum, need X * Y blocks
  const int tid = threadIdx.x;
  const int beg = tid * z / TPB;
  const int end = (tid + 1) * z / TPB;
  const int i = blockIdx.x;
  const int j = blockIdx.y;
  __shared__ T temp [32];

  // compute local sum
  T sum = 0;
  for (int k = beg; k < end; k++) {
    sum += out_3d(i, j, k);
  }

  // block-wise inclusive prefix sum
  sum = block_prefix_sum(sum, temp);

  // compute intermediate values
  for (int k = end - 1; k >= beg; k--) {
    const T val = out_3d(i, j, k);
    out_3d(i, j, k) = sum;
    sum -= val;
  }
}


static inline void d_LOR3D_i32(int& size, byte*& data, const int paramc, const double paramv [])
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
  if (paramc != 3) { 
    fprintf(stderr, "ERROR: Lorenzo 3D needs an x, y, and z as parameters\n");
    exit(-1);
  }
  const int x = (int)paramv[0];
  const int y = (int)paramv[1];
  const int z = (int)paramv[2];
  if (x * y * z != insize) {
    fprintf(stderr, "ERROR: X(%d), Y(%d), and Z(%d) don't match size %d\n", x, y, z, insize);
    exit(-1);
  }

  // Encode
  d_lorenzo_encode_3d_corner_edges_surfaces<<<std::max(y, z), std::min(std::max(x, std::max(y, z)) - 1, TPB)>>>(in_t, d_encoded, x, y, z);
  dim3 nblocks(z - 1, y - 1);
  d_lorenzo_encode_3d_interior<<<nblocks, std::min(x - 1, TPB)>>>(in_t, d_encoded, x, y);
  cudaDeviceSynchronize();

  // Finalize
  data = (byte*)d_encoded;
  cudaFree(in_t);
  return;
}


static inline void d_iLOR3D_i32(int& size, byte*& data, const int paramc, const double paramv [])
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

  // Get params
  if (paramc != 3) {
    fprintf(stderr, "ERROR: Lorenzo 3D needs an x, y, and z as parameters\n");
    exit(-1);
  }
  const int x = (int)paramv[0];
  const int y = (int)paramv[1];
  const int z = (int)paramv[2];
  if (x * y * z != insize) {
    fprintf(stderr, "ERROR: X(%d), Y(%d), and Z(%d) don't match size %d\n", x, y, z, insize);
    exit(-1);
  }

  // Decode
  dim3 x_blocks(y, z);
  dim3 y_blocks(x, z);
  dim3 z_blocks(x, y);
  d_lorenzo_decode_3d_x<<<x_blocks, TPB>>>(in_t, d_decoded, x, y);
  d_lorenzo_decode_3d_y<<<y_blocks, TPB>>>(d_decoded, x, y);
  d_lorenzo_decode_3d_z<<<z_blocks, TPB>>>(d_decoded, x, y, z);
  cudaDeviceSynchronize();

  // Finalize
  data = (byte*)d_decoded;
  cudaFree(in_t);
  return;
}