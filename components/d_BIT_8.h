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


static __device__ inline bool d_BIT_8(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int SWS = 32;  // sub-warp size
  long long* const in_l = (long long*)in;
  long long* const out_l = (long long*)out;
  const int tid = threadIdx.x;
  const int subwarp = tid / SWS;
  const int sublane = tid % SWS;
  const int extra = csize % (64 * 64 / 8);
  const int size = (csize - extra) / 8;
  assert(WS % SWS == 0);

  // works for warpsize of 64 on AMD because shfl does not include sync

  for (int i = subwarp * 64 + sublane; i < size; i += TPB * 2) {
    unsigned long long a0 = in_l[i];
    unsigned long long a1 = in_l[i + 32];

    unsigned long long b0 = a1;
    unsigned long long b1 = a0;
    unsigned long long mask = 0x00000000FFFFFFFFULL;
    a0 = (a0 & ~mask) | (b0 >> 32);
    a1 = (a1 & mask) | (b1 << 32);

    b0 = __shfl_xor_sync(~0, a0, 16);
    b1 = __shfl_xor_sync(~0, a1, 16);
    mask = 0x0000FFFF0000FFFFULL;
    if ((sublane & 16) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 16) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 16) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 16) & ~mask);
      a1 = (a1 & mask) | ((b1 << 16) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 8);
    b1 = __shfl_xor_sync(~0, a1, 8);
    mask = 0x00FF00FF00FF00FFULL;
    if ((sublane & 8) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 8) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 8) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 8) & ~mask);
      a1 = (a1 & mask) | ((b1 << 8) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 4);
    b1 = __shfl_xor_sync(~0, a1, 4);
    mask = 0x0F0F0F0F0F0F0F0FULL;
    if ((sublane & 4) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 4) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 4) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 4) & ~mask);
      a1 = (a1 & mask) | ((b1 << 4) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 2);
    b1 = __shfl_xor_sync(~0, a1, 2);
    mask = 0x3333333333333333ULL;
    if ((sublane & 2) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 2) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 2) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 2) & ~mask);
      a1 = (a1 & mask) | ((b1 << 2) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 1);
    b1 = __shfl_xor_sync(~0, a1, 1);
    mask = 0x5555555555555555ULL;
    if ((sublane & 1) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 1) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 1) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 1) & ~mask);
      a1 = (a1 & mask) | ((b1 << 1) & ~mask);
    }

    out_l[i / 64 + sublane * (size / 64)] = a0;
    out_l[i / 64 + (sublane + 32) * (size / 64)] = a1;
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  return true;
}


static __device__ inline void d_iBIT_8(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int SWS = 32;  // sub-warp size
  long long* const in_l = (long long*)in;
  long long* const out_l = (long long*)out;
  const int tid = threadIdx.x;
  const int subwarp = tid / SWS;
  const int sublane = tid % SWS;
  const int extra = csize % (64 * 64 / 8);
  const int size = (csize - extra) / 8;

  for (int i = subwarp * 64 + sublane; i < size; i += TPB * 2) {
    unsigned long long a0 = in_l[i / 64 + sublane * (size / 64)];
    unsigned long long a1 = in_l[i / 64 + (sublane + 32) * (size / 64)];

    unsigned long long b0 = a1;
    unsigned long long b1 = a0;
    unsigned long long mask = 0x00000000FFFFFFFFULL;
    a0 = (a0 & ~mask) | (b0 >> 32);
    a1 = (a1 & mask) | (b1 << 32);

    b0 = __shfl_xor_sync(~0, a0, 16);
    b1 = __shfl_xor_sync(~0, a1, 16);
    mask = 0x0000FFFF0000FFFFULL;
    if ((sublane & 16) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 16) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 16) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 16) & ~mask);
      a1 = (a1 & mask) | ((b1 << 16) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 8);
    b1 = __shfl_xor_sync(~0, a1, 8);
    mask = 0x00FF00FF00FF00FFULL;
    if ((sublane & 8) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 8) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 8) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 8) & ~mask);
      a1 = (a1 & mask) | ((b1 << 8) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 4);
    b1 = __shfl_xor_sync(~0, a1, 4);
    mask = 0x0F0F0F0F0F0F0F0FULL;
    if ((sublane & 4) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 4) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 4) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 4) & ~mask);
      a1 = (a1 & mask) | ((b1 << 4) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 2);
    b1 = __shfl_xor_sync(~0, a1, 2);
    mask = 0x3333333333333333ULL;
    if ((sublane & 2) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 2) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 2) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 2) & ~mask);
      a1 = (a1 & mask) | ((b1 << 2) & ~mask);
    }

    b0 = __shfl_xor_sync(~0, a0, 1);
    b1 = __shfl_xor_sync(~0, a1, 1);
    mask = 0x5555555555555555ULL;
    if ((sublane & 1) == 0) {
      a0 = (a0 & ~mask) | ((b0 >> 1) & mask);
      a1 = (a1 & ~mask) | ((b1 >> 1) & mask);
    } else {
      a0 = (a0 & mask) | ((b0 << 1) & ~mask);
      a1 = (a1 & mask) | ((b1 << 1) & ~mask);
    }

    out_l[i] = a0;
    out_l[i + 32] = a1;
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
}
