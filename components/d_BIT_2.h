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

static __device__ inline bool d_BIT_2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int extra = csize % (16 * 16 / 8);
  const int size = (csize - extra) / 2;
  const int tid = threadIdx.x;
  unsigned long long* const in_l = (unsigned long long*)in;
  unsigned short* const out_s = (unsigned short*)out;

  for (int pos = 16 * tid; pos < size; pos += 16 * TPB) {
    unsigned long long t, *a = &in_l[pos / 4];  // process 4 shorts in 1 long long

    for (int i = 0; i < 2; i++) {
      swp(a[i], a[i + 2], 8, 0x00FF00FF00FF00FFULL);
    }

    for (int j = 0; j < 4; j += 2) {
      swp(a[j], a[j + 1], 4, 0x0F0F0F0F0F0F0F0FULL);
    }

    for (int j = 0; j < 4; j++) {
      const unsigned long long m = 0x33333333CCCCCCCCULL;
      const unsigned long long vnm = a[j] & ~m;
      a[j] = (a[j] & m) | (vnm >> 34) | (vnm << 34);
    }

    for (int j = 0; j < 4; j++) {
      const unsigned long long m = 0x5555AAAA5555AAAAULL;
      const unsigned long long m1 = 0xFFFF0000FFFF0000ULL;
      const unsigned long long vnm = a[j] & ~m;
      const unsigned long long res = (a[j] & m) | ((vnm & m1) >> 17) | ((vnm & ~m1) << 17);
      out_s[pos / 16 + (j * 4) * (size / 16)] = res;
      out_s[pos / 16 + (j * 4 + 1) * (size / 16)] = res >> 16;
      out_s[pos / 16 + (j * 4 + 2) * (size / 16)] = res >> 32;
      out_s[pos / 16 + (j * 4 + 3) * (size / 16)] = res >> 48;
    }
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  return true;
}


static __device__ inline void d_iBIT_2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int extra = csize % (16 * 16 / 8);
  const int size = (csize - extra) / 2;
  const int tid = threadIdx.x;
  unsigned short* const in_s = (unsigned short*)in;
  unsigned long long* const out_l = (unsigned long long*)out;

  for (int pos = 16 * tid; pos < size; pos += 16 * TPB) {
    unsigned long long t, *a = &out_l[pos / 4];  // process 4 shorts in 1 long long

    for (int i = 0; i < 4; i++) {
      a[i] = in_s[pos / 16 + (i * 4) * (size / 16)] | ((unsigned long long)in_s[pos / 16 + (i * 4 + 1) * (size / 16)] << 16) | ((unsigned long long)in_s[pos / 16 + (i * 4 + 2) * (size / 16)] << 32) | ((unsigned long long)in_s[pos / 16 + (i * 4 + 3) * (size / 16)] << 48);
    }

    for (int i = 0; i < 2; i++) {
      swp(a[i], a[i + 2], 8, 0x00FF00FF00FF00FFULL);
    }

    for (int j = 0; j < 4; j += 2) {
      swp(a[j], a[j + 1], 4, 0x0F0F0F0F0F0F0F0FULL);
    }

    for (int j = 0; j < 4; j++) {
      const unsigned long long m = 0x33333333CCCCCCCCULL;
      const unsigned long long vnm = a[j] & ~m;
      a[j] = (a[j] & m) | (vnm >> 34) | (vnm << 34);
    }

    for (int j = 0; j < 4; j++) {
      const unsigned long long m = 0x5555AAAA5555AAAAULL;
      const unsigned long long m1 = 0xFFFF0000FFFF0000ULL;
      const unsigned long long vnm = a[j] & ~m;
      a[j] = (a[j] & m) | ((vnm & m1) >> 17) | ((vnm & ~m1) << 17);
    }
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
}
