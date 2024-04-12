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


#ifndef CPU_RZE
#define CPU_RZE


#include "h_zero_elimination.h"


template <typename T>
static inline bool h_RZE(int& csize, byte in [CS], byte out [CS])
{
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int extra = csize % sizeof(T);
  const int bits = 8 * sizeof(T);
  const int num = (2048 + 256 + 32 + 4) / sizeof(T);
  assert(CS == 16384);

  // zero out end of bitmap
  byte bitmap [num];
  if (csize < CS) {
    memset(&bitmap[csize / bits], 0, CS / bits - csize / bits);
  }

  // copy non-zero values and generate bitmap
  int wpos = 0;
  if (size > 0) h_ZEencode((T*)in, size, (T*)out, wpos, (T*)bitmap);
  wpos *= sizeof(T);
  if (wpos >= CS - 2 - extra) return false;

  // check if not all zeros
  if (wpos != 0) {
    int base = 0;
    int range = CS / bits;

    // iteratively compress bitmap
    while (range >= 8) {  // 2048 256 32 / sizeof(T)
      byte prev = 0;
      for (int i = 0; i < range; i += 8) {
        const long long lval = *((long long*)(&bitmap[base + i]));
        byte bmp = 0;
        for (int j = 0; j < 8; j++) {
          const byte val = (lval >> (j * 8)) & 0xff;
          if (val != prev) {
            out[wpos++] = prev = val;
            bmp |= 1 << j;
            if (wpos >= CS - 2 - extra) return false;
          }
        }
        bitmap[base + range + i / 8] = bmp;
      }
      base += range;
      range /= 8;
    }

    // output last level of bitmap
    if (wpos >= CS - 2 - extra - range) return false;
    for (int i = 0; i < range; i++) {  // 4 / sizeof(T)
      out[wpos++] = bitmap[base + i];
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    for (int i = 0; i < extra; i++) {
      out[wpos++] = in[csize - extra + i];
    }
  }

  // output old csize and update csize
  out[wpos++] = csize;  // bottom byte
  out[wpos++] = csize >> 8;  // second byte
  csize = wpos;
  return true;
}


template <typename T>
static inline void h_iRZE(int& csize, byte in [CS], byte out [CS])
{
  int rpos = csize;
  csize = (int)in[--rpos] << 8;  // second byte
  csize |= in[--rpos];  // bottom byte
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int bits = 8 * sizeof(T);
  const int num = (2048 + 256 + 32 + 4) / sizeof(T);
  assert(CS == 16384);

  // copy leftover byte
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    for (int i = 0; i < extra; i++) {
      out[csize - extra + i] = in[rpos - extra + i];
    }
    rpos -= extra;
  }

  if (rpos == 0) {
    // all zeros
    memset(out, 0, size * sizeof(T));
  } else {
    int base = 0;
    int range = CS / bits;
    while (range >= 8) {  // 2048 256 32 / sizeof(T)
      base += range;
      range /= 8;
    }

    // read in last level of bitmap
    byte bitmap [num];
    rpos -= range;
    for (int i = 0; i < range; i++) {  // 4 / sizeof(T)
      bitmap[base + i] = in[rpos + i];
    }

    // iteratively decompress bitmap
    while (range < CS / bits) {  // 32 256 2048 / sizeof(T)
      range *= 8;
      base -= range;

      if (range % 64 != 0) {
        for (int i = 0; i < range; i += 8) {
          rpos -= __builtin_popcount((int)bitmap[base + range + i / 8]);
        }
      } else {
        for (int i = 0; i < range; i += 64) {
          rpos -= __builtin_popcountll(*((long long*)(&bitmap[base + range + i / 8])));
        }
      }

      int pos = rpos;
      byte val = 0;
      for (int i = 0; i < range; i++) {
        if (bitmap[base + range + i / 8] & (1 << (i % 8))) {
          val = in[pos++];
        }
        bitmap[base + i] = val;
      }
    }

    // copy non-zero values based on bitmap
    if (size > 0) h_ZEdecode(size, (T*)in, (T*)bitmap, (T*)out);
  }
}


#endif
