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


static inline bool h_RLE_4(int& csize, byte in [CS], byte out [CS])
{
  using type = int;
  type* const in_t = (type*)in;
  type* const out_t = (type*)out;
  byte counts [CS / sizeof(type)];  // upper bound on size
  const int elems = csize / sizeof(type);
  const int extra = csize % sizeof(type);
  int cpos = 0;  // count position
  int vpos = 0;  // value position

  // look for repeating and non-repeating values
  type prev = ~in_t[0];
  int repeat = 0;
  int nrepeat = 0;
  for (int i = 0; i < elems; i++) {
    const type curr = in_t[i];
    if (prev != curr) {  // not repeating
      prev = curr;
      out_t[vpos++] = curr;  // output non-repeating value
      nrepeat++;
      // output repeat counts
      while (repeat > 0) {
        const int rep = std::min(128, repeat);
        counts[cpos++] = 0x80 | (rep - 1);
        repeat -= rep;
      }
    } else {  // repeating
      repeat++;
      // output non-repeat counts
      while (nrepeat > 0) {
        const int nrep = std::min(128, nrepeat);
        counts[cpos++] = nrep - 1;
        nrepeat -= nrep;
      }
    }
  }

  // output and remaining counts
  while (repeat > 0) {
    const int rep = std::min(128, repeat);
    counts[cpos++] = 0x80 | (rep - 1);
    repeat -= rep;
  }
  while (nrepeat > 0) {
    const int nrep = std::min(128, nrepeat);
    counts[cpos++] = nrep - 1;
    nrepeat -= nrep;
  }

  // check compressed size
  int wpos = vpos * sizeof(type);
  const int newsize = wpos + cpos + extra + 2;
  if (newsize >= CS) return false;

  // copy extra bytes at end
  for (int j = 0; j < extra; j++) {
    out[wpos++] = in[csize - extra + j];
  }

  // copy counts to output
  for (int j = 0; j < cpos; j++) {
    out[wpos + j] = counts[j];
  }

  // store position where counts start
  out[newsize - 2] = wpos & 0xff;
  out[newsize - 1] = (wpos >> 8) & 0xff;
  csize = newsize;
  return true;
}


static inline void h_iRLE_4(int& csize, byte in [CS], byte out[CS])
{
  using type = int;
  type* const in_t = (type*)in;
  type* const out_t = (type*)out;
  const int cpos = (((int)in[csize - 1]) << 8) | in[csize - 2];
  const int extra = cpos % sizeof(type);

  int wpos = 0;  // write position
  int vpos = 0;  // value position
  type val = 0;
  for (int i = cpos; i < csize - 2; i++) {
    const int rep = in[i];  // int instead of byte
    if (rep & 0x80) {
      // write repeating values
      const int repeat = (rep & 0x7f) + 1;
      for (int j = 0; j < repeat; j++) {
        out_t[wpos++] = val;
      }
    } else {
      // write non-repeating values
      const int nrepeat = rep + 1;
      for (int j = 0; j < nrepeat; j++) {
        val = in_t[vpos++];
        out_t[wpos++] = val;
      }
    }
  }

  // copy extra bytes at end
  wpos *= sizeof(type);
  for (int j = 0; j < extra; j++) {
    out[wpos++] = in[cpos - extra + j];
  }
  csize = wpos;
}
