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


static inline bool h_HCLOG_8(int& csize, byte in [CS], byte out [CS])
{
  using type = unsigned long long;  // must be unsigned
  assert(std::is_unsigned<type>::value);
  const int TB = sizeof(type) * 8;  // number of bits in type
  const int SC = 32;  // subchunks [do not change]
  const int CB = (32 - __builtin_clz(sizeof(type))) + 3;  // counter bits
  assert((1 << CB) > TB);
  assert((1 << (CB - 1)) <= TB);
  assert(SC == sizeof(int) * 8);

  // type casts
  type* in_t = (type*)in;
  type* out_t = (type*)out;
  const int size = csize / sizeof(type);

  // determine bits needed for each subchunk
  byte ln [SC];
  int flags = 0;
  int bits = 0;
  int end = 0;
  for (int i = 0; i < SC; i++) {
    // compute maximum value using both approaches
    const int beg = end;
    end = (i + 1) * size / SC;
    type max_val1 = 0;
    type max_val2 = 0;
    for (int j = beg; j < end; j++) {
      const type val1 = in_t[j];
      const type val2 = (val1 << 1) ^ (((std::make_signed_t<type>)val1) >> (sizeof(type) * 8 - 1));  // TCMS
      max_val1 = std::max(max_val1, val1);
      max_val2 = std::max(max_val2, val2);
    }

    // use approach yielding smaller max_val
    const type max_val = std::min(max_val1, max_val2);
    if (max_val2 < max_val1) {
      flags |= 1 << i;
    }

    // figure out number of bits needed
    int cnt = 0;
    if (max_val != 0) {
      cnt = (sizeof(type) == 8) ? (64 - __builtin_clzll(max_val)) : (sizeof(unsigned int) * 8 - __builtin_clz((unsigned int)max_val));
    }
    bits += cnt * (end - beg);
    ln[i] = cnt;
  }

  // check if encoded data fits
  const int extra = csize % sizeof(type);
  const int newsize = (SC + 16 + CB * SC + bits + 7) / 8;
  if (newsize + extra >= CS) return false;

  // clear out buffer
  memset(out_t, 0, newsize);

  *((int*)out) = flags;
  out[4] = csize;
  out[5] = csize >> 8;
  int loc = SC + 16;
  // encode logn values
  for (int i = 0; i < SC; i++) {
    const type val = ln[i];
    const int pos = loc / TB;
    const int shift = loc % TB;
    out_t[pos] |= val << shift;
    if (TB - CB < shift) {
      out_t[pos + 1] = val >> (TB - shift);
    }
    loc += CB;
  }

  // encode data values
  end = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = ln[i];
    const int beg = end;
    end = (i + 1) * size / SC;
    const bool flag = flags & (1 << i);
    for (int j = beg; j < end; j++) {
      type val = in_t[j];
      if (flag) {
        val = (val << 1) ^ (((std::make_signed_t<type>)val) >> (sizeof(type) * 8 - 1));  // TCMS
      }
      const int pos = loc / TB;
      const int shift = loc % TB;
      out_t[pos] |= val << shift;
      if (TB - logn < shift) {
        out_t[pos + 1] = val >> (TB - shift);
      }
      loc += logn;
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(type) > 1) {
    for (int i = 0; i < extra; i++) out[newsize + i] = in[csize - extra + i];
  }
  csize = newsize + extra;
  return true;
}


static inline void h_iHCLOG_8(int& csize, byte in [CS], byte out [CS])
{
  using type = unsigned long long;  // must be unsigned
  assert(std::is_unsigned<type>::value);
  const int TB = sizeof(type) * 8;  // number of bits in type
  const int SC = 32;  // subchunks [do not change]
  const int CB = (32 - __builtin_clz(sizeof(type))) + 3;  // counter bits
  assert((1 << CB) > TB);
  assert((1 << (CB - 1)) <= TB);

  // type casts
  type* in_t = (type*)in;
  type* out_t = (type*)out;

  // decode logn values
  const int flags = *((int*)in);
  byte ln [SC];
  int loc = SC + 16;
  const type mask = ((1 << CB) - 1);
  for (int i = 0; i < SC; i++) {
    const int pos = loc / TB;
    const int shift = loc % TB;
    type res = in_t[pos] >> shift;
    if (TB - CB < shift) {
      res |= in_t[pos + 1] << (TB - shift);
    }
    ln[i] = res & mask;
    loc += CB;
  }

  // decode data values
  const int orig_csize = in[4] + ((int)in[5] << 8);
  const int size = orig_csize / sizeof(type);
  int end = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = ln[i];
    const int beg = end;
    end = (i + 1) * size / SC;
    const type mask = (sizeof(type) < 8) ? ((1ULL << logn) - 1) : ((logn == 64) ? (~0ULL) : ((1ULL << logn) - 1));
    const bool flag = flags & (1 << i);
    for (int j = beg; j < end; j++) {
      const int pos = loc / TB;
      const int shift = loc % TB;
      type res = in_t[pos] >> shift;
      if (TB - logn < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      type val = res & mask;
      if (flag) {
        val = (val >> 1) ^ ((std::make_signed_t<type>)(val << (sizeof(type) * 8 - 1))) >> (sizeof(type) * 8 - 1);  // iTCMS
      }
      out_t[j] = val;
      loc += logn;
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(type) > 1) {
    const int extra = orig_csize % sizeof(type);
    for (int i = 0; i < extra; i++) out[orig_csize - extra + i] = in[csize - extra + i];
  }
  csize = orig_csize;
}
