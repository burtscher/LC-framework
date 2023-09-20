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


static inline void d_[name_of_preprocessor](int& size, byte*& data, const int paramc, const double paramv [])  // GPU preprocessor encoder
{
  // transforms the 'size' bytes in the 'data' array and writes the result either back to the 'data' array or to a new array and then makes 'data' point to this new array
  // if the number of bytes changes, the 'size' needs to be updated accordingly
  // the data array must start at an 8-byte aligned address
  // 'paramc' specifies the number of elements in the 'paramv' array
  // the 'paramv' array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.)
  // must be a host function that launches a kernel to do the preprocessing
  // the kernel is allowed to allocate and use shared memory
  // 'data' must be in device memory
}

static inline void d_i[name_of_preprocessor](int& size, byte*& data, const int paramc, const double paramv [])  // GPU preprocessor decoder
{
  // transforms the 'size' bytes in the 'data' array and writes the result either back to the 'data' array or to a new array and then makes 'data' point to this new array
  // if the number of bytes changes, the 'size' needs to be updated accordingly
  // the data array must start at an 8-byte aligned address
  // 'paramc' specifies the number of elements in the 'paramv' array
  // the 'paramv' array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.)
  // must be a host function that launches a kernel to do the preprocessing
  // the kernel is allowed to allocate and use shared memory
  // 'data' must be in device memory
}
