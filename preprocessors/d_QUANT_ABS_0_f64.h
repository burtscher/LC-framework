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


static __global__ void d_QUANT_ABS_0_f64_kernel(const int len, byte* const __restrict__ data, const double errorbound, const double eb2, const double inv_eb2, const double threshold)
{
  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const long long maxbin = 1LL << (mantissabits - 1);  // leave 1 bit for sign
  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    const double orig_f = data_f[idx];
    const double scaled = orig_f * inv_eb2;
    const long long bin = (long long)round(scaled);
    const double recon = bin * eb2;

    if ((bin >= maxbin) || (bin <= -maxbin) || (fabs(orig_f) >= threshold) || (recon < orig_f - errorbound) || (recon > orig_f + errorbound) || (fabs(orig_f - recon) > errorbound) || (orig_f != orig_f)) {  // last check is to handle NaNs
      assert(((data_i[idx] >> mantissabits) & 0x7ff) != 0);
    } else {
      data_i[idx] = (bin << 1) ^ (bin >> 63);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
  }
}


static __global__ void d_iQUANT_ABS_0_f64_kernel(const int len, byte* const __restrict__ data, const double eb2)
{
  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    long long bin = data_i[idx];
    if ((0 <= bin) && (bin < (1LL << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 63) >> 63));  // TCMS decoding
      data_f[idx] = bin * eb2;
    }
  }
}


static inline void d_QUANT_ABS_0_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_ABS_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  const int len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_ABS_0_f64(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const double errorbound = paramv[0];
  const double threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<double>::infinity();
  if (errorbound < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_ABS_0_f64: ERROR: error_bound must be at least %e\n", std::numeric_limits<double>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value
  if (threshold <= errorbound) {fprintf(stderr, "QUANT_ABS_0_f64: ERROR: threshold must be larger than error_bound\n"); throw std::runtime_error("LC error");}

  const double eb2 = 2 * errorbound;
  const double inv_eb2 = 0.5 / errorbound;

  d_QUANT_ABS_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, errorbound, eb2, inv_eb2, threshold);
}


static inline void d_iQUANT_ABS_0_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_ABS_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  const int len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_ABS_0_f64(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const double errorbound = paramv[0];
  if (errorbound < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_ABS_0_f64: ERROR: error_bound must be at least %e\n", std::numeric_limits<double>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value

  const double eb2 = 2 * errorbound;

  d_iQUANT_ABS_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, eb2);
}
