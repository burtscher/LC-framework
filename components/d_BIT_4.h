/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2024, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
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


static __device__ inline bool d_BIT_4(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int SWS = 32;  // sub-warp size
  int* const in_w = (int*)in;
  int* const out_w = (int*)out;
  const int tid = threadIdx.x;
  const int sublane = tid % SWS;
  const int extra = csize % (32 * 32 / 8);
  const int size = (csize - extra) / 4;
  assert(WS % SWS == 0);

  // works for warpsize of 64 on AMD because shfl does not include sync

  for (int i = tid; i < size; i += TPB) {
    unsigned int a = in_w[i];

    unsigned int q = __shfl_xor_sync(~0, a, 16);
    a = ((sublane & 16) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

    q = __shfl_xor_sync(~0, a, 8);
    a = ((sublane & 8) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

    q = __shfl_xor_sync(~0, a, 4);
    unsigned int mask = 0x0F0F0F0F;
    if ((sublane & 4) == 0) {
      a = (a & ~mask) | ((q >> 4) & mask);
    } else {
      a = ((q << 4) & ~mask) | (a & mask);
    }

    q = __shfl_xor_sync(~0, a, 2);
    mask = 0x33333333;
    if ((sublane & 2) == 0) {
      a = (a & ~mask) | ((q >> 2) & mask);
    } else {
      a = ((q << 2) & ~mask) | (a & mask);
    }

    q = __shfl_xor_sync(~0, a, 1);
    mask = 0x55555555;
    if ((sublane & 1) == 0) {
      a = (a & ~mask) | ((q >> 1) & mask);
    } else {
      a = ((q << 1) & ~mask) | (a & mask);
    }

    out_w[i / 32 + sublane * (size / 32)] = a;
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  return true;
}


static __device__ inline void d_iBIT_4(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int SWS = 32;  // sub-warp size
  int* const in_w = (int*)in;
  int* const out_w = (int*)out;
  const int tid = threadIdx.x;
  const int sublane = tid % SWS;
  const int extra = csize % (32 * 32 / 8);
  const int size = (csize - extra) / 4;
  assert(WS % SWS == 0);

  for (int i = tid; i < size; i += TPB) {
    unsigned int a = in_w[i / 32 + sublane * (size / 32)];

    unsigned int q = __shfl_xor_sync(~0, a, 16);
    a = ((sublane & 16) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

    q = __shfl_xor_sync(~0, a, 8);
    a = ((sublane & 8) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

    q = __shfl_xor_sync(~0, a, 4);
    unsigned int mask = 0x0F0F0F0F;
    if ((sublane & 4) == 0) {
      a = (a & ~mask) | ((q >> 4) & mask);
    } else {
      a = (a & mask) | ((q << 4) & ~mask);
    }

    q = __shfl_xor_sync(~0, a, 2);
    mask = 0x33333333;
    if ((sublane & 2) == 0) {
      a = (a & ~mask) | ((q >> 2) & mask);
    } else {
      a = (a & mask) | ((q << 2) & ~mask);
    }

    q = __shfl_xor_sync(~0, a, 1);
    mask = 0x55555555;
    if ((sublane & 1) == 0) {
      a = (a & ~mask) | ((q >> 1) & mask);
    } else {
      a = (a & mask) | ((q << 1) & ~mask);
    }

    out_w[i] = a;
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
}
