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


#ifndef CPU_RARE
#define CPU_RARE


template <typename T>
static inline bool h_RARE(int& csize, byte in [CS], byte out [CS])
{
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int extra = csize % sizeof(T);  // leftover bytes at end
  const int bits = 8 * sizeof(T);  // bits per word
  T* const in_t = (T*)in;  // T cast
  T* const out_t = (T*)out;

  // count how many MSBs repeat
  int count [bits + 1];  // need additional element
  for (int i = 0; i < bits; i++) count[i] = 0;  // no need to init additional element
  int zeros = 0;
  T prev = 0;
  for (int i = 0; i < size; i++) {
    const T val = in_t[i];
    const T diff = val ^ prev;
    prev = val;
    if (diff == 0) zeros++;
    const int keep = (diff == 0) ? 0 : (64 - __builtin_clzll((long long)diff));
    count[keep]++;
  }

  // special case if all values (other than extra) are zero
  if (zeros == size) {
    // copy leftover bytes
    if constexpr (sizeof(T) > 1) {
      for (int i = 0; i < extra; i++) {
        out[i] = in[csize - extra + i];
      }
    }

    // output csize and keep
    out[extra] = bits + 1;  // output special "keep" value
    out[extra + 1] = csize;  // first byte
    out[extra + 2] = csize >> 8;  // second byte
    csize = extra + 3;
    return true;
  }

  // inclusive prefix sum and find best size
  int pfs = count[0];
  int saved = bits * pfs;
  int keep = 0;
  int countk = pfs;
  for (int i = 1; i < bits; i++) {  // additional element not used
    pfs += count[i];
    const int sav = (bits - i) * pfs;
    if (saved < sav) {
      saved = sav;
      keep = i;
      countk = pfs;
    }
  }

  // special case if all bits need to be kept
  if (saved == 0) {
    // output all values without bitmap (if they fit)
    if (csize + 3 >= CS) return false;
    memcpy(out, in, csize);

    // output csize and keep
    out[csize] = bits;  // output special "keep" value
    out[csize + 1] = csize;  // first byte
    out[csize + 2] = csize >> 8;  // second byte
    csize = csize + 3;
    return true;
  }

  // keep some bits from each value (0 <= keep < bits)

  // create bitmap
  assert(CS == 16384);
  const int num = (2048 + 256 + 32 + 4) / sizeof(T);
  byte bitmap [num];

  // initialize
  const T tmask = ~(T)0 << keep;  // 111...00
  const T bmask = ~tmask;  // 000...11
  int wpos1 = 0;
  int wpos2 = size - countk;
  int wloc2 = bits * wpos2;

  // encode values and generate bitmap
  T oval = 0;
  byte bmp = 0;
  prev = 0;
  for (int i = 0; i < size; i++) {
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
          out_t[wpos2++] = oval;
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
  if ((wloc2 % bits) != 0) {
    out_t[wpos2] = oval;
  }
  if ((size % 8) != 0) {
    bitmap[size / 8] = bmp;
  }

  // zero out rest of bitmap
  for (int i = (size + 7) / 8; i < CS / bits; i++) {
    bitmap[i] = 0;
  }

  // iteratively compress bitmap
  int wpos = (wloc2 + 7) / 8;
  int base = 0;
  int range = CS / bits;
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
          if (wpos >= CS - 3 - extra) return false;
        }
      }
      bitmap[base + range + i / 8] = bmp;
    }
    base += range;
    range /= 8;
  }

  // output last level of bitmap
  if (wpos >= CS - 3 - extra - range) return false;
  for (int i = 0; i < range; i++) {
    out[wpos++] = bitmap[base + i];
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    for (int i = 0; i < extra; i++) {
      out[wpos++] = in[csize - extra + i];
    }
  }

  // output csize and keep
  out[wpos++] = keep;  // output "keep" value
  out[wpos++] = csize;  // first byte
  out[wpos++] = csize >> 8;  // second byte
  csize = wpos;
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
static inline void h_iRARE(int& csize, byte in [CS], byte out [CS])
{
  // read in original csize and keep
  const int oldsize = in[csize - 2] + ((int)in[csize - 1] << 8);
  const int keep = in[csize - 3];

  const int bits = 8 * sizeof(T);  // bits per word
  const int size = oldsize / sizeof(T);  // words in chunk (rounded down)
  const int extra = oldsize % sizeof(T);  // leftover bytes at end
  T* const in_t = (T*)in;  // T cast
  T* const out_t = (T*)out;

  if (keep == bits + 1) {  // special case
    // all values (other than extra) are zero
    memset(out, 0, oldsize - extra);
  } else if (keep == bits) {  // keep all bits
    memcpy(out, in, oldsize - extra);
  } else {  // keep some bits from each value (0 <= keep < bits)
    int base = 0;
    int range = CS / bits;
    while (range >= 8) {
      base += range;
      range /= 8;
    }

    // read in last level of bitmap
    assert(CS == 16384);
    const int num = (2048 + 256 + 32 + 4) / sizeof(T);
    byte bitmap [num];
    int count = 0;
    int rpos = csize - 3 - extra - range;
    for (int i = 0; i < range; i++) {
      const T val = in[rpos + i];
      bitmap[base + i] = val;
      count += __builtin_popcount((int)val);
    }

    // iteratively decompress bitmap
    while (range < CS / bits) {
      range *= 8;
      base -= range;
      rpos -= count;

      int pos = rpos;
      byte val = 0;
      count = 0;
      assert(range % 8 == 0);
      for (int i = 0; i < range; i += 8) {
        const byte bmp = bitmap[base + range + i / 8];
        for (int j = 0; j < 8; j++) {
          if ((bmp >> j) & 1) {
            val = in[pos++];
          }
          bitmap[base + i + j] = val;
          count += __builtin_popcount((int)val);
        }
      }
    }

    // decode values
    const T tmask = ~(T)0 << keep;  // 111...00
    const T bmask = ~tmask;  // 000...11
    int rpos1 = 0;
    int rloc2 = bits * count;
    int rpos2 = rloc2 / bits;
    T ival = in_t[rpos2++];
    byte bmp;
    T prev = 0, val = 0;
    for (int i = 0; i < size; i++) {
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
    for (int i = 0; i < extra; i++) {
      out[oldsize - extra + i] = in[csize - 3 - extra + i];
    }
  }

  csize = oldsize;
}


#endif
