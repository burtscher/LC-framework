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


#ifndef zero_elimination_host
#define zero_elimination_host


template <typename T, bool check = false>
static inline bool h_ZEencode(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout)  // all sizes in number of words
{
  const int bits = sizeof(T) * 8;  // bits per word
  const int num = (insize + bits - 1) / bits;  // number of subchunks (rounded up)
  int pos = 0;
  int cnt = 0;
  for (int i = 0; i < num - 1; i++) {
    T bm = 0;
    for (int j = 0; j < bits; j++) {
      const T val = in[cnt++];
      if (val != 0) {
        if constexpr (check) {
          if (pos >= datasize) return false;
        }
        bm |= (T)1 << j;
        dataout[pos++] = val;
      }
    }
    bmout[i] = bm;
  }
  const int i = num - 1;
  {
    T bm = 0;
    for (int j = 0; j < bits; j++) {
      if (cnt >= insize) break;
      const T val = in[cnt++];
      if (val != 0) {
        if constexpr (check) {
          if (pos >= datasize) return false;
        }
        bm |= (T)1 << j;
        dataout[pos++] = val;
      }
    }
    bmout[i] = bm;
  }
  datasize = pos;
  return true;
}


template <typename T>
static inline void h_ZEdecode(const int decsize, const T* const datain, const T* const bmin, T* const out)  // all sizes in number of words
{
  const int bits = sizeof(T) * 8;  // bits per word
  const int num = (decsize + bits - 1) / bits;  // number of subchunks (rounded up)
  int pos = 0;
  int cnt = 0;
  for (int i = 0; i < num - 1; i++) {
    const T bm = bmin[i];
    for (int j = 0; j < bits; j++) {
      T val = 0;
      if (((bm >> j) & 1) != 0) {
        val = datain[pos++];
      }
      out[cnt++] = val;
    }
  }
  const int i = num - 1;
  {
    const T bm = bmin[i];
    for (int j = 0; j < bits; j++) {
      T val = 0;
      if (((bm >> j) & 1) != 0) {
        val = datain[pos++];
      }
      out[cnt++] = val;
      if (cnt >= decsize) break;
    }
  }
}


#endif
