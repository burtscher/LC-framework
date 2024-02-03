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


#include <cstdlib>
#include <cstdio>
#if defined(_OPENMP)
#include <omp.h>
#endif


static inline void h_LOR1D_i32(int& size, byte*& data, const int paramc, const double paramv [])
{
  using type = int;
  type* in_t = (type*)data;
  // Check that size is correct
  if (size % sizeof(type) != 0) {
    fprintf(stderr, "ERROR: size %d is not evenly divisible by type size %ld\n", size, sizeof(type));
    exit(-1);
  }
  int insize = size / sizeof(type);
  type* out = new type[insize];

  // Encode
  if (insize > 0) {
    out[0] = in_t[0];
    #pragma omp parallel for default(none) shared(out, in_t, insize)
    for (int i = 1; i < insize; i++) {
      out[i] = in_t[i] - in_t[i - 1];
    }
  }

  // Finalize
  delete [] data;
  data = (byte*)out;
  return;
}


static inline void h_iLOR1D_i32(int& size, byte*& data, const int paramc, const double paramv [])
{
  using type = int;
  type* in_t = (type*)data;
  // Check that size is correct
  if (size % sizeof(type) != 0) {
    fprintf(stderr, "ERROR: size %d is not evenly divisible by type size %ld\n", size, sizeof(type));
    exit(-1);
  }
  int insize = size / sizeof(type);
  type* out = new type[insize];

  // Decode
  #if defined(_OPENMP)
  type* rightmost = new type [omp_get_max_threads()];
  #pragma omp parallel default(none) shared(insize, in_t, out, rightmost)
  {
    const int nthreads = omp_get_num_threads();
    const int my_rank = omp_get_thread_num();
    const int my_start = my_rank * (long)insize / nthreads;
    const int my_end = (my_rank + 1) * (long)insize / nthreads;

    // Perform local prefix sum
    type sum = 0;
    for (int i = my_start; i < my_end; i++) {
      sum += in_t[i];
      out[i] = sum;
    }
    // Store rightmost for rest following chunks to use
    rightmost[my_rank] = sum;

    // Barrier so every thread has completed their local
    #pragma omp barrier

    // Calculate correction
    type correction = 0;
    for (int i = 0; i < my_rank; i++) {
      correction += rightmost[i];
    }
    if (correction != 0) {
      for (int i = my_start; i < my_end; i++) {
        out[i] += correction;
      }
    }
  }
  // Delete rightmost inside the if
  delete [] rightmost;
  #else
  type sum = 0;
  for (int i = 0; i < insize; i++) {
    sum += in_t[i];
    out[i] = sum;
  }
  #endif

  // Finalize
  delete [] data;
  data = (byte*)out;
  return;
}