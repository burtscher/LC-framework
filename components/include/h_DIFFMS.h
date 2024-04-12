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


#ifndef CPU_DIFFMS
#define CPU_DIFFMS


template <typename T>
static inline bool h_DIFFMS(int& csize, byte in [CS], byte out [CS])
{
  T* const in_t = (T*)in;  // T cast
  T* const out_t = (T*)out;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)

  // compute difference sequence plus TCMS
  T prev = 0;
  for (int i = 0; i < size; i++) {
    const T val = in_t[i];
    const T data = val - prev;
    prev = val;
    out_t[i] = (data << 1) ^ ((std::make_signed_t<T>)data) >> (sizeof(T) * 8 - 1);
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    for (int i = 0; i < extra; i++) {
      out[csize - extra + i] = in[csize - extra + i];
    }
  }
  return true;
}


template <typename T>
static inline void h_iDIFFMS(int& csize, byte in [CS], byte out [CS])
{
  T* const in_t = (T*)in;  // T cast
  T* const out_t = (T*)out;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)

  // compute prefix sum
  T sum = 0;
  for (int i = 0; i < size; i++) {
    const T data = in_t[i];
    const T val = (data >> 1) ^ ((std::make_signed_t<T>)(data << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);
    sum += val;
    out_t[i] = sum;
  }

  // copy leftover byte
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    for (int i = 0; i < extra; i++) {
      out[csize - extra + i] = in[csize - extra + i];
    }
  }
}


#endif
