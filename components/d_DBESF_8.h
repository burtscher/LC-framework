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


static __device__ inline bool d_DBESF_8(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  using type = unsigned long long;
  const int size = csize / sizeof(type);
  const int extra = csize % sizeof(type);
  type* const bufin = (type*)in;
  type* const bufout = (type*)out;
  const int tid = threadIdx.x;

  // convert from IEEE 754 to ESF
  for (int i = tid; i < size; i += TPB) {
    const type val = bufin[i];
    const type f = ((1ULL << 52) - 1) & val;
    const type e = val >> 52;  // extracts S and E, S shifted out later
    const type v = (e << 1) | (val >> 63);
    bufout[i] = ((v << 52) | f) - 0x7fe0'0000'0000'0000;
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  return true;
}


static __device__ inline void d_iDBESF_8(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  using type = unsigned long long;
  const int size = csize / sizeof(type);
  const int extra = csize % sizeof(type);
  type* const bufin = (type*)in;
  type* const bufout = (type*)out;
  const int tid = threadIdx.x;

  // convert from ESF to IEEE 754
  for (int i = tid; i < size; i += TPB) {
    const type val = bufin[i] + 0x7fe0'0000'0000'0000;
    const type s = val >> 52;  // extracts E and S, E shifted out later
    const type f = ((1ULL << 52) - 1) & val;
    const type v = (s << 11) | (val >> 53);
    bufout[i] = (v << 52) | f;
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
}
