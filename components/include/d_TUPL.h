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


#ifndef GPU_TUPL
#define GPU_TUPL


template <typename T, int dim>
static __device__ inline bool d_TUPL(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  assert(dim > 1);
  const int tid = threadIdx.x;
  const int tuples = csize / sizeof(T) / dim;  // tuples in chunk (rounded down)
  const int size = tuples * dim;  // words in all tuples
  const int extra = csize - sizeof(T) * size;  // leftover bytes
  T* const buf_in = (T*)in;  // type cast
  T* const buf_out = (T*)out;  // type cast

  // copy leftover bytes
  if (tid < extra) out[sizeof(T) * size + tid] = in[sizeof(T) * size + tid];

  // distribute fields of structs
  for (int i = tid; i < size; i += TPB) {
    buf_out[(i / dim) + (i % dim) * tuples] = buf_in[i];
  }
  return true;
}


template <typename T, int dim>
static __device__ inline void d_iTUPL(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  assert(dim > 1);
  const int tid = threadIdx.x;
  const int tuples = csize / sizeof(T) / dim;  // tuples in chunk (rounded down)
  const int size = tuples * dim;  // words in all tuples
  const int extra = csize - sizeof(T) * size;  // leftover bytes
  T* const buf_in = (T*)in;  // type cast
  T* const buf_out = (T*)out;  // type cast

  // copy leftover bytes
  if (tid < extra) out[sizeof(T) * size + tid] = in[sizeof(T) * size + tid];

  // recombine fields of structs
  for (int i = tid; i < size; i += TPB) {
    buf_out[i] = buf_in[(i / dim) + (i % dim) * tuples];
  }
}


#endif
