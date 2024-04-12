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


#ifndef GPU_CLOG
#define GPU_CLOG


template <typename T>
static __device__ inline bool d_CLOG(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;

  assert(std::is_unsigned<T>::value);
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int SC = 32;  // subchunks [do not change]
  const int CB = (32 - __clz((int)sizeof(T))) + 3;  // counter bits
  assert((1 << CB) > TB);
  assert((1 << (CB - 1)) <= TB);
  assert(WS >= SC);
  assert(SC == sizeof(int) * 8);
  assert(sizeof(int) == 4);

  // T casts
  T* const in_t = (T*)in;
  int* const out_i = (int*)out;  // int
  const int TB_i = sizeof(int) * 8;
  const int size = csize / sizeof(T);

  byte* const ln = temp;
  int* const bits = (int*)&temp[SC];
  int* const total_bits = (int*)&bits[WS];

  // determine bits needed for each subchunk
  for (int i = warp; i < SC; i += warps) {
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;

    T max_val = 0;
    // max of values for each thread
    for (int j = beg + lane; j < end; j += WS) {
      max_val = max(max_val, in_t[j]);
    }

    // warp level max
    max_val = max(max_val, __shfl_xor_sync(~0, max_val, 1));
    max_val = max(max_val, __shfl_xor_sync(~0, max_val, 2));
    max_val = max(max_val, __shfl_xor_sync(~0, max_val, 4));
    max_val = max(max_val, __shfl_xor_sync(~0, max_val, 8));
    max_val = max(max_val, __shfl_xor_sync(~0, max_val, 16));
#if defined(WS) && (WS == 64)
    max_val = max(max_val, __shfl_xor_sync(~0, max_val, 32));
#endif

    // use approach yielding smaller max_val
    if (lane == 0) {
      // figure out number of bits needed
      int cnt = 0;
      if (max_val != 0) {
        cnt = (sizeof(T) == 8) ? (64 - __clzll(max_val)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val));
      }
      bits[i] = cnt * (end - beg); // total bits of each chunk
      ln[i] = cnt;  // logn value for each chunk
    }
  }
  __syncthreads();

  // warp prefix sum over bits
  if (warp == 0) {
    const int org = bits[lane];
    int val = org;
    int tmp = __shfl_up_sync(~0, val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up_sync(~0, val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up_sync(~0, val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up_sync(~0, val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up_sync(~0, val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
    if (lane == SC - 1) *total_bits = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int extra = csize % sizeof(T);
  const int newsize = (16 + CB * SC + *total_bits + 7) / 8;
  if (newsize + extra >= CS) return false;

  // clear out buffer
  for (int i = tid; i < (newsize + sizeof(int) - 1) / sizeof(int); i += TPB) out_i[i] = 0;
  __syncthreads();

  // encode logn values
  if (lane < SC) {
    const int val = ln[lane];
    const int loc = 16 + (CB * lane);
    const int pos = loc / TB_i;
    const int shift = loc % TB_i;
    atomicOr_block(&out_i[pos], val << shift);
    if (TB_i - CB < shift) {
      atomicOr_block(&out_i[pos + 1], val >> (TB_i - shift));
    }
  }

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = ln[i];
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;
    const int offs = 16 + CB * SC + bits[i];
    for (int j = beg + lane; j < end; j += WS) {
      const T val = in_t[j];
      const int loc = offs + (j - beg) * logn;
      if constexpr (sizeof(T) < 8) {
        const int pos = loc / TB_i;
        const int shift = loc % TB_i;
        atomicOr_block(&out_i[pos], (unsigned int)val << shift);
        if (TB_i - logn < shift) {
          atomicOr_block(&out_i[pos + 1], (unsigned int)val >> (TB_i - shift));
        }
      } else {
        long long* const out_l = (long long*)out;
        const int pos = loc / TB;
        const int shift = loc % TB;
        atomicOr_block(&out_l[pos], val << shift);
        if (TB - logn < shift) {
          atomicOr_block(&out_l[pos + 1], val >> (TB - shift));
        }
      }
    }
  }
  __syncthreads();

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[newsize + tid] = in[csize - extra + tid];
  }

  // record old csize
  if (tid == 0) {
    *((short*)out) = csize;
  }
  csize = newsize + extra;
  return true;
}


template <typename T>
static __device__ inline void d_iCLOG(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;

  assert(std::is_unsigned<T>::value);
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int SC = 32;  // subchunks [do not change]
  const int CB = (32 - __clz((int)sizeof(T))) + 3;  // counter bits
  assert((1 << CB) > TB);
  assert((1 << (CB - 1)) <= TB);
  assert(WS >= SC);

  // T casts
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  byte* const ln = (byte*)temp;
  int* const bits = (int*)&temp[SC];
  const int orig_csize = *((short*)in);
  const int size = orig_csize / sizeof(T);

  // decode logn values
  const T mask = ((1 << CB) - 1);
  if (warp == 0) {
    T res = 0;
    if (lane < SC) {
      const int loc = 16 + (lane * CB);
      const int pos = loc / TB;
      const int shift = loc % TB;
      res = in_t[pos] >> shift;
      if (TB - CB < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      res &= mask;
      ln[lane] = res;
    }

    const int beg = lane * size / SC;
    const int end = (lane + 1) * size / SC;
    const int org = res * (end - beg);
    int val = org;
    int tmp = __shfl_up_sync(~0, val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up_sync(~0, val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up_sync(~0, val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up_sync(~0, val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up_sync(~0, val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
  }
  __syncthreads();

  // decode data values
  for (int i = warp; i < SC; i += TPB / WS) {
    const int logn = ln[i];
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;
    const T mask = (sizeof(T) < 8) ? ((1ULL << logn) - 1) : ((logn == 64) ? (~0ULL) : ((1ULL << logn) - 1));
    const int offs = 16 + SC * CB + bits[i];
    for (int j = beg + lane; j < end; j += WS) {
      const int loc = offs + (j - beg) * logn;
      const int pos = loc / TB;
      const int shift = loc % TB;
      T res = in_t[pos] >> shift;
      if (TB - logn < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      out_t[j] = res & mask;
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    const int extra = orig_csize % sizeof(T);
    if (tid < extra) out[orig_csize - extra + tid] = in[csize - extra + tid];
  }
  csize = orig_csize;
}


#endif
