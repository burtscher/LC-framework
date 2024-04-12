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


#ifndef max_reduction
#define max_reduction

template <typename T>
static __device__ inline T block_max_reduction(T val, void* buffer)  // returns max to all threads
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;
  T* const s_carry = (T*)buffer;
  assert(WS >= warps);

  val = max(val, __shfl_xor_sync(~0, val, 1));
  val = max(val, __shfl_xor_sync(~0, val, 2));
  val = max(val, __shfl_xor_sync(~0, val, 4));
  val = max(val, __shfl_xor_sync(~0, val, 8));
  val = max(val, __shfl_xor_sync(~0, val, 16));
#if defined(WS) && (WS == 64)
  val = max(val, __shfl_xor_sync(~0, val, 32));
#endif
  if (lane == 0) s_carry[warp] = val;
  __syncthreads();  // s_carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      val = (lane < warps) ? s_carry[lane] : 0;
      val = max(val, __shfl_xor_sync(~0, val, 1));
      if constexpr (warps > 2) {
        val = max(val, __shfl_xor_sync(~0, val, 2));
        if constexpr (warps > 4) {
          val = max(val, __shfl_xor_sync(~0, val, 4));
          if constexpr (warps > 8) {
            val = max(val, __shfl_xor_sync(~0, val, 8));
            if constexpr (warps > 16) {
              val = max(val, __shfl_xor_sync(~0, val, 16));
              #if defined(WS) && (WS == 64)
              if constexpr (warps > 32) {
                val = max(val, __shfl_xor_sync(~0, val, 32));
              }
              #endif
            }
          }
        }
      }
      s_carry[lane] = val;
    }
    __syncthreads();  // s_carry updated
  }

  return s_carry[0];
}

#endif
