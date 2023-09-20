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


#include <cstdlib>
#include <cstdio>


#ifndef in_3d_h
#define in_3d_h(a, b, c) in_t[((c) * y + (b)) * x + (a)]
#endif


#ifndef out_3d_h
#define out_3d_h(a, b, c) out[((c) * y + (b)) * x + (a)]
#endif


static inline void h_LOR3D_i32(int& size, byte*& data, const int paramc, const double paramv [])
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

  // Get params
  if (paramc != 3) {
    fprintf(stderr, "ERROR: Lorenzo 3D needs x, y, and z as parameters\n");
    exit(-1);
  }
  const int x = (int)paramv[0];
  const int y = (int)paramv[1];
  const int z = (int)paramv[2];
  if (x * y * z != insize) {
    fprintf(stderr, "ERROR: X(%d), Y(%d), and Z(%d) don't match size %d\n", x, y, z, insize);
    exit(-1);
  }

  // Encode
  if (insize > 0) {
    // Compute edge elements
    out_3d_h(0, 0, 0) = in_3d_h(0, 0, 0);
    #pragma omp parallel default(none) shared(in_t, out, x, y, z)
    {
      #pragma omp for nowait
      for (int i = 1; i < x; i++) {
        out_3d_h(i, 0, 0) = in_3d_h(i, 0, 0) - in_3d_h(i - 1, 0, 0);
      }
      #pragma omp for nowait
      for (int j = 1; j < y; j++) {
        out_3d_h(0, j, 0) = in_3d_h(0, j, 0) - in_3d_h(0, j - 1, 0);
      }
      #pragma omp for nowait
      for (int k = 1; k < z; k++) {
        out_3d_h(0, 0, k) = in_3d_h(0, 0, k) - in_3d_h(0, 0, k - 1);
      }

      // Surface elements
      #pragma omp for nowait
      for (int j = 1; j < y; j++) {
        for (int i = 1; i < x; i++) {
          out_3d_h(i, j, 0) = in_3d_h(i, j, 0) - in_3d_h(i - 1, j, 0) - in_3d_h(i, j - 1, 0) + in_3d_h(i - 1, j - 1, 0);
        }
      }
      #pragma omp for nowait
      for (int k = 1; k < z; k++) {
        for (int i = 1; i < x; i++) {
          out_3d_h(i, 0, k) = in_3d_h(i, 0, k) - in_3d_h(i - 1, 0, k) - in_3d_h(i, 0, k - 1) + in_3d_h(i - 1, 0, k - 1);
        }
      }
      #pragma omp for nowait
      for (int k = 1; k < z; k++) {
        for (int j = 1; j < y; j++) {
          out_3d_h(0, j, k) = in_3d_h(0, j, k) - in_3d_h(0, j - 1, k) - in_3d_h(0, j, k - 1) + in_3d_h(0, j - 1, k - 1);
        }
      }

      // Internal elements
      #pragma omp for nowait
      for (int k = 1; k < z; k++) {
        for (int j = 1; j < y; j++) {
          for (int i = 1; i < x; i++) {
            out_3d_h(i, j, k) = in_3d_h(i, j, k) - in_3d_h(i - 1, j, k) - in_3d_h(i, j - 1, k) - in_3d_h(i, j, k - 1) + in_3d_h(i - 1, j - 1, k) + in_3d_h(i - 1, j, k - 1) + in_3d_h(i, j - 1, k - 1) - in_3d_h(i - 1, j - 1, k - 1);
          }
        }
      }
    }
  }

  // Finalize
  delete [] data;
  data = (byte*)out;
  return;
}


static inline void h_iLOR3D_i32(int& size, byte*& data, const int paramc, const double paramv [])
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

  // Get params
  if (paramc != 3) {
    fprintf(stderr, "ERROR: Lorenzo 3D needs x, y, and z as parameters\n");
    exit(-1);
  }
  const int x = (int)paramv[0];
  const int y = (int)paramv[1];
  const int z = (int)paramv[2];
  if (x * y * z != insize) {
    fprintf(stderr, "ERROR: X(%d), Y(%d), and Z(%d) don't match size %d\n", x, y, z, insize);
    exit(-1);
  }

  // Decode
  #pragma omp parallel default(none) shared(in_t, out, x, y, z)
  {
    // x-direction prefix sums
    #pragma omp for
    for (int k = 0; k < z; k++) {
      for (int j = 0; j < y; j++) {
        type prev = out_3d_h(0, j, k) = in_3d_h(0, j, k);
        for (int i = 1; i < x; i++) {
          const type val = in_3d_h(i, j, k) + prev;
          out_3d_h(i, j, k) = val;
          prev = val;
        }
      }
    }

    // y-direction prefix sums
    #pragma omp for
    for (int k = 0; k < z; k++) {
      for (int i = 0; i < x; i++) {
        for (int j = 1; j < y; j++) {
          out_3d_h(i, j, k) = out_3d_h(i, j, k) + out_3d_h(i, j - 1, k);
        }
      }
    }

    // z-direction prefix sums
    #pragma omp for
    for (int j = 0; j < y; j++) {
      for (int i = 0; i < x; i++) {
        for (int k = 1; k < z; k++) {
          out_3d_h(i, j, k) = out_3d_h(i, j, k) + out_3d_h(i, j, k - 1);
        }
      }
    }
  }

  // Finalize
  delete [] data;
  data = (byte*)out;
  return;
}