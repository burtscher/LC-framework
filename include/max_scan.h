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


#ifndef max_scan
#define max_scan

template <typename T>
static __device__ inline T block_max_scan(T val, void* buffer)  // returns inclusive maximum scan
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;
  T* const carry = (T*)buffer;
  assert(WS >= warps);

  T tmp = __shfl_up_sync(~0, val, 1);
  if (lane >= 1) val = max(val, tmp);
  tmp = __shfl_up_sync(~0, val, 2);
  if (lane >= 2) val = max(val, tmp);
  tmp = __shfl_up_sync(~0, val, 4);
  if (lane >= 4) val = max(val, tmp);
  tmp = __shfl_up_sync(~0, val, 8);
  if (lane >= 8) val = max(val, tmp);
  tmp = __shfl_up_sync(~0, val, 16);
  if (lane >= 16) val = max(val, tmp);
#if defined(WS) && (WS == 64)
  tmp = __shfl_up_sync(~0, val, 32);
  if (lane >= 32) val = max(val, tmp);
#endif

  if (lane == WS - 1) carry[warp] = val;
  __syncthreads();  // carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      T res = carry[lane];
      T tmp = __shfl_up_sync(~0, res, 1);
      if (lane >= 1) res = max(res, tmp);
      if constexpr (warps > 2) {
        tmp = __shfl_up_sync(~0, res, 2);
        if (lane >= 2) res = max(res, tmp);
        if constexpr (warps > 4) {
          tmp = __shfl_up_sync(~0, res, 4);
          if (lane >= 4) res = max(res, tmp);
          if constexpr (warps > 8) {
            tmp = __shfl_up_sync(~0, res, 8);
            if (lane >= 8) res = max(res, tmp);
            if constexpr (warps > 16) {
              tmp = __shfl_up_sync(~0, res, 16);
              if (lane >= 16) res = max(res, tmp);
              #if defined(WS) && (WS == 64)
              if constexpr (warps > 32) {
                tmp = __shfl_up_sync(~0, res, 32);
                if (lane >= 32) res = max(res, tmp);
              }
              #endif
            }
          }
        }
      }
      carry[lane] = res;
    }
    __syncthreads();  // carry updated

    if (warp > 0) val = max(val, carry[warp - 1]);
  }

  return val;
}

#endif
