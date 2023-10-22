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


static __host__ __device__ inline double d_QUANT_REL_0_f64_log2approx(const double orig_f)
{
  const int mantissabits = 52;
  const long long orig_i = *((long long*)&orig_f);
  const int expo = (orig_i >> mantissabits) & 0x7ff;
  const long long frac_i = (1023LL << mantissabits) | (orig_i & ~(~0LL << mantissabits));
  const double frac_f = *((double*)&frac_i);
  const double log_f = frac_f + (expo - 1024);  // - bias - 1
  return log_f;
}


static __device__ inline double d_QUANT_REL_0_f64_pow2approx(const double log_f)
{
  const int mantissabits = 52;
  const double biased = log_f + 1023;
  const long long expo = biased;
  const double frac_f = biased - (expo - 1);
  const long long frac_i = *((long long*)&frac_f);
  const long long exp_i = (expo << mantissabits) | (frac_i & ~(~0LL << mantissabits));
  const double recon_f = *((double*)&exp_i);
  return recon_f;
}


static __global__ void d_QUANT_REL_0_f64_kernel(const int len, byte* const __restrict__ data, const double errorbound, const double log2eb, const double inv_log2eb, const double threshold)
{
  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const long long signexpomask = ~0LL << mantissabits;
  const long long maxbin = (1LL << (mantissabits - 2)) - 1;  // leave 2 bits for 2 signs (plus one element)
  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    const long long orig_i = data_i[idx];
    const long long abs_orig_i = orig_i & 0x7fff'ffff'ffff'ffffLL;
    const double abs_orig_f = *((double*)&abs_orig_i);
    long long output = orig_i;
    const int expo = (orig_i >> mantissabits) & 0x7ff;
    if (expo == 0) {  // zero or de-normal values
      if (abs_orig_i == 0) {  // zero
        output = signexpomask | 1;
      }
    } else {
      if (expo == 0x7ff) {  // INF or NaN
        if (((orig_i & signexpomask) == signexpomask) && ((orig_i & ~signexpomask) != 0)) {  // negative NaN
          output = abs_orig_i;  // make positive NaN
        }
      } else {  // normal value
        const double log_f = d_QUANT_REL_0_f64_log2approx(abs_orig_f);
        const double scaled = log_f * inv_log2eb;
        long long bin = (long long)roundf(scaled);
        const double abs_recon_f = d_QUANT_REL_0_f64_pow2approx(bin * log2eb);
        const double lower = abs_orig_f / (1 + errorbound);
        const double upper = abs_orig_f * (1 + errorbound);
        if (!((bin >= maxbin) || (bin <= -maxbin) || (abs_orig_f >= threshold) || (abs_recon_f < lower) || (abs_recon_f > upper) || (abs_recon_f == 0) || !isfinite(abs_recon_f))) {
          bin = (bin << 1) ^ (bin >> 63);  // TCMS encoding
          bin = (bin + 1) << 1;
          if (orig_i < 0) bin |= 1;  // include sign
          output = signexpomask | bin;  // 'sign' and 'exponent' fields are all ones, 'mantissa' is non-zero (looks like a negative NaN)
        }
      }
    }
    data_i[idx] = (output ^ signexpomask) - 1;
  }
}


static __global__ void d_iQUANT_REL_0_f64_kernel(const int len, byte* const __restrict__ data, const double log2eb)
{
  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const long long signexpomask = ~0LL << mantissabits;
  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    const long long val = (data_i[idx] + 1) ^ signexpomask;
    if (((val & signexpomask) == signexpomask) && ((val & ~signexpomask) != 0)) {  // is encoded value
      if (val == (signexpomask | 1)) {
        data_i[idx] = 0;
      } else {
        const long long dec = ((val & ~signexpomask) >> 1) - 1;
        const long long bin = (dec >> 1) ^ (((dec << 63) >> 63));  // TCMS decoding
        const double abs_recon_f = d_QUANT_REL_0_f64_pow2approx(bin * log2eb);
        data_f[idx] = (val & 1) ? -abs_recon_f : abs_recon_f;
      }
    } else {
      data_i[idx] = val;
    }
  }
}


static inline void d_QUANT_REL_0_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_REL_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); exit(-1);}
  const int len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_REL_0_f64(error_bound [, threshold])\n"); exit(-1);}
  const double errorbound = paramv[0];
  const double threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<double>::infinity();
  if (errorbound < 1E-7) {fprintf(stderr, "QUANT_REL_0_f64: ERROR: error_bound must be at least %e\n", 1E-7); exit(-1);}  // minimum positive normalized value
  if (threshold <= errorbound) {fprintf(stderr, "QUANT_REL_0_f64: ERROR: threshold must be larger than error_bound\n"); exit(-1);}

  const double log2eb = 2 * d_QUANT_REL_0_f64_log2approx(1 + errorbound);
  const double inv_log2eb = 1 / log2eb;

  d_QUANT_REL_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, errorbound, log2eb, inv_log2eb, threshold);
}


static inline void d_iQUANT_REL_0_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_REL_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); exit(-1);}
  const int len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_REL_0_f64(error_bound [, threshold])\n"); exit(-1);}
  const double errorbound = paramv[0];
  if (errorbound < 1E-7) {fprintf(stderr, "QUANT_REL_0_f64: ERROR: error_bound must be at least %e\n", 1E-7); exit(-1);}  // minimum positive normalized value

  const double log2eb = 2 * d_QUANT_REL_0_f64_log2approx(1 + errorbound);

  d_iQUANT_REL_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, log2eb);
}
