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


#ifndef CPU_CLOG
#define CPU_CLOG


template <typename T>
static inline bool h_CLOG(int& csize, byte in [CS], byte out [CS])
{
  assert(std::is_unsigned<T>::value);
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int SC = 32;  // subchunks
  const int CB = (32 - __builtin_clz(sizeof(T))) + 3;  // counter bits
  assert((1 << CB) > TB);
  assert((1 << (CB - 1)) <= TB);

  // T casts
  T* in_t = (T*)in;
  T* out_t = (T*)out;
  const int size = csize / sizeof(T);

  // determine bits needed for each subchunk
  byte ln [SC];
  int bits = 0;
  int end = 0;
  for (int i = 0; i < SC; i++) {
    const int beg = end;
    end = (i + 1) * size / SC;
    T max_val = 0;
    for (int j = beg; j < end; j++) {
      max_val = std::max(max_val, in_t[j]);
    }
    int cnt = 0;
    if (max_val != 0) {
      cnt = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __builtin_clzll((unsigned long long)max_val)) : (sizeof(unsigned int) * 8 - __builtin_clz((unsigned int)max_val));
    }
    bits += cnt * (end - beg);
    ln[i] = cnt;
  }

  // check if encoded data fits
  const int extra = csize % sizeof(T);
  const int newsize = (16 + CB * SC + bits + 7) / 8;
  if (newsize + extra >= CS) return false;

  // clear out buffer
  memset(out_t, 0, newsize);

  out[0] = csize & 0xFF;
  out[1] = csize >> 8;
  int loc = 16;
  // encode logn values
  for (int i = 0; i < SC; i++) {
    const T val = ln[i];
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
    for (int j = beg; j < end; j++) {
      const T val = in_t[j];
      const int pos = loc / TB;
      const int shift = loc % TB;
      out_t[pos] |= val << shift;
      if (TB - logn < shift) {
        out_t[pos + 1] = val >> (TB - shift);
      }
      loc += logn;
    }
  }

  // copy extra bytes at end and update csize
  for (int i = 0; i < extra; i++) out[newsize + i] = in[csize - extra + i];
  csize = newsize + extra;
  return true;
}


template <typename T>
static inline void h_iCLOG(int& csize, byte in [CS], byte out [CS])
{
  assert(std::is_unsigned<T>::value);
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int SC = 32;  // subchunks
  const int CB = (32 - __builtin_clz(sizeof(T))) + 3;  // counter bits
  assert((1 << CB) > TB);
  assert((1 << (CB - 1)) <= TB);

  // T casts
  T* in_t = (T*)in;
  T* out_t = (T*)out;

  // decode logn values
  byte ln [SC];
  int loc = 16;
  const T mask = ((1 << CB) - 1);
  for (int i = 0; i < SC; i++) {
    const int pos = loc / TB;
    const int shift = loc % TB;
    T res = in_t[pos] >> shift;
    if (TB - CB < shift) {
      res |= in_t[pos + 1] << (TB - shift);
    }
    ln[i] = res & mask;
    loc += CB;
  }

  // decode data values
  const int orig_csize = *((unsigned short*)in);
  const int size = orig_csize / sizeof(T);
  int end = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = ln[i];
    const int beg = end;
    end = (i + 1) * size / SC;
    const T mask = (sizeof(T) < 8) ? ((1ULL << logn) - 1) : ((logn == 64) ? (~0ULL) : ((1ULL << logn) - 1));
    for (int j = beg; j < end; j++) {
      const int pos = loc / TB;
      const int shift = loc % TB;
      T res = in_t[pos] >> shift;
      if (TB - logn < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      out_t[j] = res & mask;
      loc += logn;
    }
  }

  // copy extra bytes at end and update csize
  const int extra = orig_csize % sizeof(T);
  for (int i = 0; i < extra; i++) out[orig_csize - extra + i] = in[csize - extra + i];
  csize = orig_csize;
}


#endif
