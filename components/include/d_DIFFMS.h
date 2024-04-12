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


#ifndef GPU_DIFFMS
#define GPU_DIFFMS


template <typename T>
static __device__ inline bool d_DIFFMS(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int tid = threadIdx.x;

  // compute difference sequence
  for (int i = tid; i < size; i += TPB) {
    const T prev = (i == 0) ? 0 : in_t[i - 1];
    const T val = in_t[i];
    const T data = val - prev;
    out_t[i] = (data << 1) ^ ((std::make_signed_t<T>)data) >> (sizeof(T) * 8 - 1);
  }

  // copy leftover bytes at end
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  }
  return true;
}


template <typename T>
static __device__ inline void d_iDIFFMS(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int tid = threadIdx.x;
  const int beg = tid * size / TPB;
  const int end = (tid + 1) * size / TPB;

  // compute local sums
  T sum = 0;
  for (int i = beg; i < end; i++) {
    const T data = in_t[i];
    const T val = (data >> 1) ^ ((std::make_signed_t<T>)(data << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);
    sum += val;
    in_t[i] = val;
  }

  // compute prefix sum
  sum = block_prefix_sum(sum, temp);  // includes barrier

  // compute intermediate values
  for (int i = end - 1; i >= beg; i--) {
    out_t[i] = sum;
    sum -= in_t[i];
  }

  // copy leftover bytes at end
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  }
}


#endif
