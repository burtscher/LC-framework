/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
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


#ifndef LC_MACROS_H
#define LC_MACROS_H


/*
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
  #define __kernel__ __global__ __launch_bounds__(TPB, 2048 / TPB)
#else
  #define __kernel__ __global__ __launch_bounds__(TPB, 1536 / TPB)
#endif
*/


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
static inline __device__ int __reduce_add_sync(const int mask, int val)
{
  val += __shfl_xor_sync(~0, val, 1);
  val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 4);
  val += __shfl_xor_sync(~0, val, 8);
  val += __shfl_xor_sync(~0, val, 16);
  return val;
}
#endif


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  #define __all(arg) __all_sync(~0, arg)
  #define __any(arg) __any_sync(~0, arg)
  #define __ballot(arg) __ballot_sync(~0, arg)
  #define __shfl(...) __shfl_sync(~0, __VA_ARGS__)
  #define __shfl_down(...) __shfl_down_sync(~0, __VA_ARGS__)
  #define __shfl_up(...) __shfl_up_sync(~0, __VA_ARGS__)
  #define __shfl_xor(...) __shfl_xor_sync(~0, __VA_ARGS__)
#endif


#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static inline __device__ void __syncwarp() {}
  #define __trap() abort()
  #define atomicOr_block(...) atomicOr(__VA_ARGS__)
  #define atomicAdd_block(...) atomicAdd(__VA_ARGS__)
  namespace cuda::std { using ::std::numeric_limits; }
#endif

#endif  /* LC_MACROS_H */
