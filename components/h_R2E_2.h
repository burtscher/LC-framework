/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2023, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
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


#include "include/h_repetition_elimination.h"


static inline bool h_R2E_2(int& csize, byte in [CS], byte out [CS])
{
  using type = short;
  const int bits = sizeof(type) * 8;
  const type* const in_t = (type*)in;  // type cast
  type* const out_t = (type*)out;  // type cast
  const int num = (csize / sizeof(type) + bits - 1) / bits;  // number of subchunks (rounded up)
  const int extra = csize % sizeof(type);
  type bitmap [CS / sizeof(type) / bits];

  // compute bitmap and copy non-repeating values
  int pos;
  h_REencode(in_t, csize / sizeof(type), out_t, pos, bitmap);
  pos *= sizeof(type);

  // compress bitmap
  const int numb = (num * sizeof(type) + 8 - 1) / 8;  // number of subchunks (rounded up)
  int loc = CS - 4 - extra - (pos + numb);
  if (loc <= 0) return false;
  if (!h_REencode<byte, true>((byte*)bitmap, num * sizeof(type), &out[pos + numb], loc, &out[pos])) return false;
  loc += pos + numb;

  // copy leftover bytes at end
  for (int i = 0; i < extra; i++) {
    out[loc++] = in[csize - extra + i];
  }

  // record pos
  out[loc++] = pos & 0xff;
  out[loc++] = (pos >> 8) & 0xff;

  // record csize
  out[loc++] = csize & 0xff;
  out[loc++] = (csize >> 8) & 0xff;

  csize = loc;
  return true;
}


static inline void h_iR2E_2(int& csize, byte in [CS], byte out [CS])
{
  // get csize
  using type = short;
  const int bits = sizeof(type) * 8;
  const type* const in_t = (type*)in;  // type cast
  type* const out_t = (type*)out;
  const int old = csize;
  const int pos = (((int)in[csize - 3]) << 8) | in[csize - 4];
  csize = (((int)in[csize - 1]) << 8) | in[csize - 2];
  const int extra = csize % sizeof(type);  // extra bytes at end
  type bitmap [CS / sizeof(type) / bits];

  // decompress bitmap
  const int num = (csize / sizeof(type) + bits - 1) / bits;  // number of subchunks (rounded up)
  const int numb = (num * sizeof(type) + 8 - 1) / 8;  // number of subchunks (rounded up)
  h_REdecode(num * sizeof(type), &in[pos + numb], &in[pos], (byte*)bitmap);

  // copy non-repeating values based on bitmap
  h_REdecode(csize / sizeof(type), in_t, bitmap, out_t);

  // copy leftover bytes
  for (int i = 0; i < extra; i++) {
    out[csize - extra + i] = in[old - 4 - extra + i];
  }
}
