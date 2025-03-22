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


#ifndef GPU_RARE
#define GPU_RARE


#include "d_zero_elimination.h"
#include "d_repetition_elimination.h"


template <typename T>
static __device__ inline bool d_RARE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int extra = csize % sizeof(T);
  const int bits = 8 * sizeof(T);
  assert(CS == 16384);
  T* const in_t = (T*)in;  // T cast
  T* const out_t = (T*)out;

  // count how many MSBs repeat
  int* const count = (int*)temp;  // bits + 2 elements (66)
  if (tid < bits) count[tid] = 0;
  __syncthreads();

  bool allzeros = true;
  for (int i = tid; i < size; i += TPB) {
    const T prev = (i > 0) ? in_t[i - 1] : 0;
    const T val = in_t[i] ^ prev;
    if (val != 0) allzeros = false;
    if constexpr (sizeof(T) == 8) {
      const int keep = (val == 0) ? 0 : (64 - __builtin_clzll((long long)val));
      atomicAdd_block(&count[keep], 1);
    } else {
      const int keep = (val == 0) ? 0 : (32 - __builtin_clz((int)val));
      atomicAdd_block(&count[keep], 1);
    }
  }
  allzeros = __syncthreads_and(allzeros);

  // special case if all values (other than extra) are zero
  if (allzeros) {
    // copy leftover bytes
    if constexpr (sizeof(T) > 1) {
      if (tid < extra) out[tid] = in[csize - extra + tid];
    }

    // output csize and keep
    if (tid == WS) {
      out[extra] = bits + 1;  // output special "keep" value
      out[extra + 1] = csize;  // first byte
      out[extra + 2] = csize >> 8;  // second byte
    }
    csize = extra + 3;
    return true;
  }

  // prefix sum and find best size
  if constexpr (bits <= WS) {
    if (tid < WS) {  // first warp only
      // prefix sum of counts
      const int lane = tid;
      int pfs = count[lane];
      int tmp = __shfl_up(pfs, 1);
      if (lane >= 1) pfs += tmp;
      tmp = __shfl_up(pfs, 2);
      if (lane >= 2) pfs += tmp;
      tmp = __shfl_up(pfs, 4);
      if (lane >= 4) pfs += tmp;
      if constexpr (bits > 8) {
        tmp = __shfl_up(pfs, 8);
        if (lane >= 8) pfs += tmp;
        if constexpr (bits > 16) {
          tmp = __shfl_up(pfs, 16);
          if (lane >= 16) pfs += tmp;
          #if defined(WS) && (WS == 64)
          if constexpr (bits > 32) {
            tmp = __shfl_up(pfs, 32);
            if (lane >= 32) pfs += tmp;
          }
          #endif
        }
      }
      count[lane] = pfs;

      // determine maximum savings
      const int sav = (bits <= lane) ? -1 : ((bits - lane) * pfs);
      int val = sav;
      val = max(val, __shfl_xor(val, 1));
      val = max(val, __shfl_xor(val, 2));
      val = max(val, __shfl_xor(val, 4));
      val = max(val, __shfl_xor(val, 8));
      val = max(val, __shfl_xor(val, 16));
      #if defined(WS) && (WS == 64)
      val = max(val, __shfl_xor(val, 32));
      const long long bal = __ballot(val == sav);
      const int who = __ffsll(bal) - 1;
      #else
      static_assert(WS == 32);
      const int bal = __ballot(val == sav);
      const int who = __ffs(bal) - 1;
      #endif
      if (lane == 0) count[64] = val;  // saved
      if (lane == 0) count[65] = who;  // keep
    }
  } else {
    assert(bits == WS * 2);
    if (tid < WS) {  // first warp only
      const int l0 = tid * 2;
      const int l1 = l0 + 1;

      // prefix sum of counts
      const int lane = tid;
      const int c1 = count[l1];
      int pfs = count[l0] + c1;
      int tmp = __shfl_up(pfs, 1);
      if (lane >= 1) pfs += tmp;
      tmp = __shfl_up(pfs, 2);
      if (lane >= 2) pfs += tmp;
      tmp = __shfl_up(pfs, 4);
      if (lane >= 4) pfs += tmp;
      if constexpr (bits > 8) {
        tmp = __shfl_up(pfs, 8);
        if (lane >= 8) pfs += tmp;
        if constexpr (bits > 16) {
          tmp = __shfl_up(pfs, 16);
          if (lane >= 16) pfs += tmp;
          #if defined(WS) && (WS == 64)
          if constexpr (bits > 32) {
            tmp = __shfl_up(pfs, 32);
            if (lane >= 32) pfs += tmp;
          }
          #endif
        }
      }
      count[l1] = pfs;
      count[l0] = pfs - c1;

      // determine maximum savings
      const int sav1 = (bits - l1) * pfs;
      const int sav0 = (bits - l0) * (pfs - c1);
      int val = max(sav0, sav1);
      val = max(val, __shfl_xor(val, 1));
      val = max(val, __shfl_xor(val, 2));
      val = max(val, __shfl_xor(val, 4));
      val = max(val, __shfl_xor(val, 8));
      val = max(val, __shfl_xor(val, 16));
      #if defined(WS) && (WS == 64)
      val = max(val, __shfl_xor(val, 32));
      const long long bal = __ballot((val == sav0) || (val == sav1));
      const int who = __ffsll(bal) - 1;
      #else
      static_assert(WS == 32);
      const int bal = __ballot((val == sav0) || (val == sav1));
      const int who = __ffs(bal) - 1;
      #endif
      if (lane == who) {
        count[64] = val;  // saved
        count[65] = (val == sav0) ? l0 : l1;  // keep
      }
    }
  }
  __syncthreads();

  const int saved = count[64];
  const int keep = count[65];
  const int countk = count[keep];

  // special case if all bits need to be kept
  if (saved == 0) {
    // output all values without bitmap (if they fit)
    if (csize + 3 >= CS) return false;
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = in_t[i];
    }

    // copy leftover bytes
    if constexpr (sizeof(T) > 1) {
      if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
    }

    // output csize and keep
    if (tid == 0) {
      out[csize] = bits;  // output special "keep" value
      out[csize + 1] = csize;  // first byte
      out[csize + 2] = csize >> 8;  // second byte
    }
    csize += 3;
    return true;
  }

  // keep some bits from each value (0 <= keep < bits)

  //zero out for atomic OR
  for (int i = tid + size - countk; i < size - countk + ((countk * keep + bits - 1) / bits); i += TPB) {
    out_t[i] = 0;
  }
  __syncthreads();

  // create bitmap
  byte* const bitmap = (byte*)&count[66];  // num elements

  // initialize
  const T tmask = ~(T)0 << keep;  // 111...00
  const T bmask = ~tmask;  // 000...11

  // determine wpos1
  const int ept = (((size + TPB - 1) / TPB + 7) / 8) * 8;  // elements per thread (multiple of 8)
  int cnt = 0;
  T prev = ((tid * ept == 0) || (tid * ept >= size)) ? 0 : (in_t[tid * ept - 1] & tmask);
  for (int i = tid * ept; i < min((tid + 1) * ept, size); i++) {
    const T val = in_t[i];
    if (prev != (val & tmask)) {
      prev = val & tmask;
      cnt++;
    }
  }
  int wpos1 = block_prefix_sum(cnt, temp) - cnt;
  int wloc2 = bits * (size - countk) + (tid * ept - wpos1) * keep;
  int wpos2 = wloc2 / bits;

  // encode values and generate bitmap
  T oval = 0;
  byte bmp = 0;
  prev = ((tid * ept == 0) || (tid * ept >= size)) ? 0 : (in_t[tid * ept - 1] & tmask);
  for (int i = tid * ept; i < min((tid + 1) * ept, size); i++) {
    const T val = in_t[i];
    if (prev != (val & tmask)) {
      prev = val & tmask;
      bmp |= 1 << (i % 8);
      out_t[wpos1++] = val;  // output all bits
    } else {
      if (keep != 0) {
        // output bottom bits only
        const T bval = val & bmask;
        const int shift = wloc2 % bits;
        const int bms = bits - shift;
        oval |= bval << shift;
        if (bms <= keep) {
          out_t[wpos2++] = oval;  //atomicOr_block(&out_t[wpos2++], oval);
          oval = bval >> bms;
        }
        wloc2 += keep;
      }
    }
    if ((i % 8) == 7) {
      bitmap[i / 8] = bmp;
      bmp = 0;
    }
  }
  if ((tid * ept < size) && ((tid + 1) * ept > size)) {
    bitmap[size / 8] = bmp;
  }

  // zero out rest of bitmap
  for (int i = tid + (size + 7) / 8; i < CS / bits; i += TPB) {
    bitmap[i] = 0;
  }
  __syncthreads();

  // output last partial word
  if ((wloc2 % bits) != 0) {
    if constexpr (bits == 8) {
      atomicOr_block((int*)&out_t[wpos2 & ~3], (int)oval << (8 * (wpos2 & 3)));
    } else if constexpr (bits == 16) {
      atomicOr_block((int*)&out_t[wpos2 & ~1], (int)oval << (16 * (wpos2 & 1)));
    } else if constexpr (bits == 32) {
      atomicOr_block((int*)&out_t[wpos2], (int)oval);
    } else {
      atomicOr_block((unsigned long long*)&out_t[wpos2], (unsigned long long)oval);
    }      
  }

  // iteratively compress bitmaps
  const int avail = CS - 3 - extra;
  int wpos = (bits * (size - countk) + keep * countk + 7) / 8;
  int base = 0 / sizeof(T);
  int range = 2048 / sizeof(T);
  cnt = avail - wpos;
  if (!d_REencode<byte, 2048 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], (int*)temp)) return false;
  wpos += cnt;
  __syncthreads();

  base = 2048 / sizeof(T);
  range = 256 / sizeof(T);
  cnt = avail - wpos;
  if (!d_REencode<byte, 256 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], (int*)temp)) return false;
  wpos += cnt;
  __syncthreads();

  base = (2048 + 256) / sizeof(T);
  range = 32 / sizeof(T);
  if constexpr (sizeof(T) < 8) {
    cnt = avail - wpos;
    if (!d_REencode<byte, 32 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], (int*)temp)) return false;
    wpos += cnt;

    base = (2048 + 256 + 32) / sizeof(T);
    range = 4 / sizeof(T);
  }

  // output last level of bitmap
  if (wpos >= avail - range) return false;
  if (tid < range) {  // 4 / sizeof(T)
    out[wpos + tid] = bitmap[base + tid];
  }
  wpos += range;

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[wpos + tid] = in[csize - extra + tid];
  }

  // output old csize and update csize
  const int new_size = wpos + 3 + extra;
  if (tid == 0) {
    out[new_size - 3] = keep;  // output "keep" value
    out[new_size - 2] = csize;  // bottom byte
    out[new_size - 1] = csize >> 8;  // second byte
  }
  csize = new_size;
  return true;
}


/*
  bits*(size-count[keep]): full values
  keep*count[keep]: bottoms
  ?: compressed bitmap
  possible padding
  (csize%4)*8: extra
  8: keep
  16: csize
*/


template <typename T>
static __device__ inline void d_iRARE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  // read in original csize and keep
  const int oldsize = in[csize - 2] + ((int)in[csize - 1] << 8);
  const int keep = in[csize - 3];

  const int tid = threadIdx.x;
  const int bits = 8 * sizeof(T);  // bits per word
  const int size = oldsize / sizeof(T);  // words in chunk (rounded down)
  const int extra = oldsize % sizeof(T);  // leftover bytes at end
  T* const in_t = (T*)in;  // T cast
  T* const out_t = (T*)out;
  assert(TPB >= 256);

  if (keep == bits + 1) {  // special case
    // all values (other than extra) are zero
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = 0;
    }
  } else if (keep == bits) {  // keep all bits
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = in_t[i];
    }
  } else {  // keep some bits from each value (0 <= keep < bits)
    assert(CS == 16384);
    int rpos = csize - 3 - extra;
    int* const temp_w = (int*)temp;
    byte* const bitmap = (byte*)&temp_w[WS];

    // iteratively decompress bitmaps
    int base, range;
    if constexpr (sizeof(T) == 8) {
      base = (2048 + 256) / sizeof(T);
      range = 32 / sizeof(T);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];
    } else {
      base = (2048 + 256 + 32) / sizeof(T);
      range = 4 / sizeof(T);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];

      rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
      base = (2048 + 256) / sizeof(T);
      range = 32 / sizeof(T);
      d_REdecode<byte, 32 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    d_REdecode<byte, 256 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    if constexpr (sizeof(T) >= 4) {
      rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    }
    if constexpr (sizeof(T) == 2) {
      int sum = __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
      sum += __syncthreads_count((tid + TPB < range * 8) && ((bitmap[base + (tid + TPB) / 8] >> (tid % 8)) & 1));
      rpos -= sum;
    }
    if constexpr (sizeof(T) == 1) {
      int sum = 0;
      for (int i = 0; i < TPB * 4; i += TPB) {
        sum += __syncthreads_count((tid + i < range * 8) && ((bitmap[base + (tid + i) / 8] >> (tid % 8)) & 1));
      }
      rpos -= sum;
    }
    base = 0 / sizeof(T);
    range = 2048 / sizeof(T);
    d_REdecode<byte, 2048 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // determine rpos1 etc.
    const int ept = (((size + TPB - 1) / TPB + 7) / 8) * 8;  // elements per thread (multiple of 8)
    int cnt = 0;
    for (int i = tid * ept; i < min((tid + 1) * ept, size); i += 8) {
      cnt += __builtin_popcount((int)bitmap[i / 8]);
    }
    int rpos1 = block_prefix_sum(cnt, temp_w) - cnt;
    const int count = temp_w[TPB / WS - 1];
    int rloc2 = bits * count + (tid * ept - rpos1) * keep;
    int rpos2 = rloc2 / bits;

    // decode values
    const T tmask = ~(T)0 << keep;  // 111...00
    const T bmask = ~tmask;  // 000...11
    T ival = in_t[rpos2++];
    byte bmp;
    T val = (rpos1 > 0) ? in_t[rpos1 - 1] : 0;
    T prev = val & tmask;
    for (int i = tid * ept; i < min((tid + 1) * ept, size); i++) {
      if ((i % 8) == 0) bmp = bitmap[i / 8];
      if ((bmp >> (i % 8)) & 1) {
        val = in_t[rpos1++];  // read all bits
        prev = val & tmask;
      } else {
        if (keep != 0) {
          // read only bottom bits
          const int shift = rloc2 % bits;
          const int bms = bits - shift;
          T res = ival >> shift;
          if (bms <= keep) {
            ival = in_t[rpos2++];
            res |= ival << bms;
          }
          rloc2 += keep;
          const T bot = res & bmask;
          val = prev | bot;
        }
      }
      out_t[i] = val;
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) {
      out[oldsize - extra + tid] = in[csize - 3 - extra + tid];
    }
  }

  csize = oldsize;
}


#endif
