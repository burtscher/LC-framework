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


#ifndef GPU_RLE
#define GPU_RLE


template <typename T>
static __device__ inline bool d_RLE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  int* const s_tmp = (int*)temp;
  const int elems = csize / (int)sizeof(T);
  const int extra = csize % (int)sizeof(T);
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;
  const int beg = tid * elems / TPB;
  const int end = (tid + 1) * elems / TPB;
  assert(CS / (int)sizeof(T) / TPB <= (int)sizeof(int) * 8);

  // count non-repeating values and perform local max scan
  const T first = (beg == 0) ? (~in_t[0]) : in_t[beg - 1];
  int bmp = 0;
  int msk = 0;
  int loc = -1;
  int nrepeat = 0;
  T prev = first;
  T curr = (beg < end) ? in_t[beg] : 0;
  for (int i = beg; i < end; i++) {
    const T next = (i + 1 < elems) ? in_t[i + 1] : 0;
    const int shft = 1 << (i - beg);
    if (prev != curr) {
      nrepeat++;
      msk |= shft;
    }
    if (((prev == curr) != (curr == next)) || (i + 1 == elems)) {
      loc = i;
      bmp |= shft;
    }
    prev = curr;
    curr = next;
  }

  // compute inclusive prefix sum and make exclusive
  int vpos = block_prefix_sum(nrepeat, &s_tmp[WS]) - nrepeat;

  // compute inclusive max scan and make exclusive (with -1)
  const int msres = block_max_scan(loc, &s_tmp[0]);
  int prevmax = __shfl_up_sync(~0, msres, 1);
  if (lane == 0) prevmax = (warp == 0) ? -1 : s_tmp[warp - 1];

  // output non-repeating values and determine number of counters
  int cnt = 0;
  int prior = prevmax;
  prev = first;
  for (int i = beg; i < end; i++) {
    const T curr = in_t[i];
    if (prev != curr) {
      out_t[vpos++] = curr;
    }
    prev = curr;
    if (bmp & (1 << (i - beg))) {
      const int dist = i - prior;
      prior = i;
      cnt += (dist + 127) / 128;
    }
  }

  // compute prefix sum
  int cpos = block_prefix_sum(cnt, &s_tmp[WS * 2]);
  const int tot = vpos * (int)sizeof(T) + extra + cpos + 2;
  if (tid == TPB - 1) {
    s_tmp[WS * 3] = vpos * (int)sizeof(T) + extra;
    s_tmp[WS * 3 + 1] = tot;
  }
  if (__syncthreads_or(tot >= CS)) return false;

  // copy leftover bytes
  const int wpos = s_tmp[WS * 3];
  if (tid < extra) {
    out[wpos - extra + tid] = in[csize - extra + tid];
  }
  csize = s_tmp[WS * 3 + 1];

  // write counts to output
  cpos -= cnt;
  prior = prevmax;
  for (int i = beg; i < end; i++) {
    const int shft = 1 << (i - beg);
    if (bmp & shft) {
      const byte mask = (msk & shft) ? 0x00 : 0x80;
      int dist = i - prior;
      prior = i;
      while (dist > 0) {
        const int rep = min(128, dist);
        out[wpos + cpos++] = (mask | (rep - 1));
        dist -= rep;
      }
    }
  }

  // store position where counts start
  if (tid == TPB - 1) {
    out[csize - 2] = (wpos & 0xff);
    out[csize - 1] = ((wpos >> 8) & 0xff);
  }
  return true;
}


template <typename T>
static __device__ inline void d_iRLE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  const int cpos = (((int)in[csize - 1]) << 8) | in[csize - 2];
  const int extra = cpos % (int)sizeof(T);
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;
  const int warps = TPB / WS;

  int rpos = 0;
  int wpos = 0;
  int bot = cpos;
  int top = (cpos & ~(warps - 1)) + warp;
  if (top < cpos) top += warps;

  while (bot < csize - 2) {
    int rsum = 0;
    int wsum = 0;
    int pos = top - WS + lane;
    if ((bot <= pos) && (pos < csize - 2)) {
      const int rep = in[pos];  // int instead of byte
      const int repeat = (rep & 0x7f) + 1;
      wsum += repeat;
      if ((rep & 0x80) == 0) rsum += repeat;
    }
    rsum += __shfl_xor_sync(~0, rsum, 1);
    rsum += __shfl_xor_sync(~0, rsum, 2);
    rsum += __shfl_xor_sync(~0, rsum, 4);
    rsum += __shfl_xor_sync(~0, rsum, 8);
    rsum += __shfl_xor_sync(~0, rsum, 16);
#if defined(WS) && (WS == 64)
    rsum += __shfl_xor_sync(~0, rsum, 32);
#endif
    wsum += __shfl_xor_sync(~0, wsum, 1);
    wsum += __shfl_xor_sync(~0, wsum, 2);
    wsum += __shfl_xor_sync(~0, wsum, 4);
    wsum += __shfl_xor_sync(~0, wsum, 8);
    wsum += __shfl_xor_sync(~0, wsum, 16);
#if defined(WS) && (WS == 64)
    wsum += __shfl_xor_sync(~0, wsum, 32);
#endif
    rpos += rsum;
    wpos += wsum;

    if (top < csize - 2) {
      const int rep = in[top];  // int instead of byte
      if (rep & 0x80) {
        // write repeating values
        const int repeat = (rep & 0x7f) + 1;
        const T val = in_t[rpos - 1];
        for (int j = lane; j < repeat; j += WS) {
          out_t[wpos + j] = val;
        }
      } else {
        // write non-repeating values
        const int nrepeat = rep + 1;
        for (int j = lane; j < nrepeat; j += WS) {
          out_t[wpos + j] = in_t[rpos + j];
        }
      }
    }

    bot = top;
    top += warps;
  }

  // copy leftover bytes
  csize = wpos * (int)sizeof(T) + extra;
  if (tid < extra) out[csize - extra + tid] = in[cpos - extra + tid];
}


#endif
