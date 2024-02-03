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

static inline bool h_BIT_4(int& csize, byte in [CS], byte out [CS])
{
  const int extra = csize % (32 * 32 / 8);
  const int size = (csize - extra) / 4;
  unsigned long long* const in_l = (unsigned long long*)in;
  unsigned int* const out_w = (unsigned int*)out;

  for (int pos = 0; pos < size; pos += 32) {
    unsigned long long t, *a = &in_l[pos / 2];  // process 2 ints in 1 long long

    for (int i = 0; i < 8; i++) {
      swp(a[i], a[i + 8], 16, 0x0000FFFF0000FFFFULL);
    }

    for (int j = 0; j < 16; j += 8) {
      for (int i = 0; i < 4; i++) {
        swp(a[j + i], a[j + i + 4], 8, 0x00FF00FF00FF00FFULL);
      }
    }

    for (int j = 0; j < 16; j += 4) {
      for (int i = 0; i < 2; i++) {
        swp(a[j + i], a[j + i + 2], 4, 0x0F0F0F0F0F0F0F0FULL);
      }
    }

    for (int j = 0; j < 16; j += 2) {
      swp(a[j], a[j + 1], 2, 0x3333333333333333ULL);
    }

    for (int j = 0; j < 16; j++) {
      const unsigned long long m = 0x55555555AAAAAAAAULL;
      const unsigned long long vnm = a[j] & ~m;
      const unsigned long long res = (a[j] & m) | (vnm >> 33) | (vnm << 33);
      out_w[pos / 32 + (j * 2) * (size / 32)] = res;
      out_w[pos / 32 + (j * 2 + 1) * (size / 32)] = res >> 32;
    }
  }

  // copy leftover bytes
  for (int i = 0; i < extra; i++) {
    out[csize - extra + i] = in[csize - extra + i];
  }
  return true;
}


static inline void h_iBIT_4(int& csize, byte in [CS], byte out [CS])
{
  const int extra = csize % (32 * 32 / 8);
  const int size = (csize - extra) / 4;
  unsigned int* const in_w = (unsigned int*)in;
  unsigned long long* const out_l = (unsigned long long*)out;

  for (int pos = 0; pos < size; pos += 32) {
    unsigned long long t, *a = &out_l[pos / 2];  // process 2 ints in 1 long long

    for (int i = 0; i < 8; i++) {
      const unsigned long long s1 = in_w[pos / 32 + (i * 2) * (size / 32)] | ((unsigned long long)in_w[pos / 32 + (i * 2 + 1) * (size / 32)] << 32);
      const unsigned long long s2 = in_w[pos / 32 + (i * 2 + 16) * (size / 32)] | ((unsigned long long)in_w[pos / 32 + (i * 2 + 1 + 16) * (size / 32)] << 32);
      const unsigned long long t = (s1 ^ (s2 >> 16)) & 0x0000FFFF0000FFFFULL;
      a[i] = s1 ^ t;
      a[i + 8] = s2 ^ (t << 16);
    }

    for (int j = 0; j < 16; j += 8) {
      for (int i = 0; i < 4; i++) {
        swp(a[j + i], a[j + i + 4], 8, 0x00FF00FF00FF00FFULL);
      }
    }

    for (int j = 0; j < 16; j += 4) {
      for (int i = 0; i < 2; i++) {
        swp(a[j + i], a[j + i + 2], 4, 0x0F0F0F0F0F0F0F0FULL);
      }
    }

    for (int j = 0; j < 16; j += 2) {
      swp(a[j], a[j + 1], 2, 0x3333333333333333ULL);
    }

    for (int j = 0; j < 16; j++) {
      const unsigned long long m = 0x55555555AAAAAAAAULL;
      const unsigned long long vnm = a[j] & ~m;
      a[j] = (a[j] & m) | (vnm >> 33) | (vnm << 33);
    }
  }

  // copy leftover bytes
  for (int i = 0; i < extra; i++) {
    out[csize - extra + i] = in[csize - extra + i];
  }
}
