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


#define swp(x, y, s, m) t = ((x) ^ ((y) >> (s))) & (m);  (x) ^= t;  (y) ^= t << (s);

static inline bool h_BIT_8(int& csize, byte in [CS], byte out [CS])
{
  const int extra = csize % (64 * 64 / 8);
  const int size = (csize - extra) / 8;
  unsigned long long* const in_l = (unsigned long long*)in;
  unsigned long long* const out_l = (unsigned long long*)out;

  for (int pos = 0; pos < size; pos += 64) {
    unsigned long long t, *a = &in_l[pos];

    for (int i = 0; i < 32; i++) {
      swp(a[i], a[i + 32], 32, 0x00000000FFFFFFFFULL);
    }

    for (int j = 0; j < 64; j += 32) {
      for (int i = 0; i < 16; i++) {
        swp(a[j + i], a[j + i + 16], 16, 0x0000FFFF0000FFFFULL);
      }
    }

    for (int j = 0; j < 64; j += 16) {
      for (int i = 0; i < 8; i++) {
        swp(a[j + i], a[j + i + 8], 8, 0x00FF00FF00FF00FFULL);
      }
    }

    for (int j = 0; j < 64; j += 8) {
      for (int i = 0; i < 4; i++) {
        swp(a[j + i], a[j + i + 4], 4, 0x0F0F0F0F0F0F0F0FULL);
      }
    }

    for (int j = 0; j < 64; j += 4) {
      for (int i = 0; i < 2; i++) {
        swp(a[j + i], a[j + i + 2], 2, 0x3333333333333333ULL);
      }
    }

    for (int j = 0; j < 64; j += 2) {
      t = (a[j] ^ (a[j + 1] >> 1)) & 0x5555555555555555ULL;
      out_l[pos / 64 + j * (size / 64)] = a[j] ^ t;
      out_l[pos / 64 + (j + 1) * (size / 64)] = a[j + 1] ^ (t << 1);
    }
  }

  // copy leftover bytes
  for (int i = 0; i < extra; i++) {
    out[csize - extra + i] = in[csize - extra + i];
  }
  return true;
}


static inline void h_iBIT_8(int& csize, byte in [CS], byte out [CS])
{
  const int extra = csize % (64 * 64 / 8);
  const int size = (csize - extra) / 8;
  unsigned long long* const in_l = (unsigned long long*)in;
  unsigned long long* const out_l = (unsigned long long*)out;

  for (int pos = 0; pos < size; pos += 64) {
    unsigned long long t, *a = &out_l[pos];

    for (int i = 0; i < 32; i++) {
      const long long ai = in_l[pos / 64 + i * (size / 64)];
      const long long ai32 = in_l[pos / 64 + (i + 32) * (size / 64)];
      t = (ai ^ (ai32 >> 32)) & 0x00000000FFFFFFFFULL;
      a[i] = ai ^ t;
      a[i + 32] = ai32 ^ (t << 32);
    }

    for (int j = 0; j < 64; j += 32) {
      for (int i = 0; i < 16; i++) {
        swp(a[j + i], a[j + i + 16], 16, 0x0000FFFF0000FFFFULL);
      }
    }

    for (int j = 0; j < 64; j += 16) {
      for (int i = 0; i < 8; i++) {
        swp(a[j + i], a[j + i + 8], 8, 0x00FF00FF00FF00FFULL);
      }
    }

    for (int j = 0; j < 64; j += 8) {
      for (int i = 0; i < 4; i++) {
        swp(a[j + i], a[j + i + 4], 4, 0x0F0F0F0F0F0F0F0FULL);
      }
    }

    for (int j = 0; j < 64; j += 4) {
      for (int i = 0; i < 2; i++) {
        swp(a[j + i], a[j + i + 2], 2, 0x3333333333333333ULL);
      }
    }

    for (int j = 0; j < 64; j += 2) {
      swp(a[j], a[j + 1], 1, 0x5555555555555555ULL);
    }
  }

  // copy leftover bytes
  for (int i = 0; i < extra; i++) {
    out[csize - extra + i] = in[csize - extra + i];
  }
}
