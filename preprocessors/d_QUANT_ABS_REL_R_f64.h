/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2026, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
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


// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ unsigned int d_QUANT_ABS_REL_R_f64_hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


static __host__ __device__ inline double d_QUANT_ABS_REL_R_f64_log2approxf(const double orig_f)
{
  //assert(orig_f > 0);
  const int mantissabits = 52;
  const long long orig_i = *((long long*)&orig_f);
  const int expo = (orig_i >> mantissabits) & 0x7ff;
  //if ((expo == 0) || (expo == 0x7ff)) return orig_f;
  const long long frac_i = (1023LL << mantissabits) | (orig_i & ~(~0LL << mantissabits));
  const double frac_f = *((double*)&frac_i);
  const double log_f = frac_f + (expo - 1024);  // - bias - 1
  return log_f;
}


static __device__ inline double d_QUANT_ABS_REL_R_f64_pow2approxf(const double log_f)
{
  const int mantissabits = 52;
  const double biased = log_f + 1023LL;
  const long long expo = biased;
  const double frac_f = biased - (expo - 1);
  const long long frac_i = *((long long*)&frac_f);
  const long long exp_i = (expo << mantissabits) | (frac_i & ~(~0LL << mantissabits));
  const double recon_f = *((double*)&exp_i);
  return recon_f;
}


static __global__ void d_QUANT_ABS_REL_R_f64_kernel(const long long len, byte* const __restrict__ data, const double abs_errorbound, const double rel_errorbound, const double eb, const double inv_eb, const double log2eb, const double inv_log2eb, const double threshold, const double boundary, const double bnd_log_f, const double bnd_scaled, const double bnd_bin)
{
  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const long long mask = (1LL << mantissabits) - 1;
  const double inv_mask = 1.0 / mask;
  const long long signexpomask = ~0LL << mantissabits;
  const long long maxbin = (1LL << (mantissabits - 2)) - 1;  // leave 2 bits for 2 signs (plus one element)
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    const long long orig_i = data_i[idx];
    const long long abs_orig_i = orig_i & 0x7fff'ffff'ffff'ffffLL;
    const double abs_orig_f = *((double*)&abs_orig_i);

    long long output = orig_i;
    const long long expo = (orig_i >> mantissabits) & 0x7ff;
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
        if (abs_orig_f >= boundary) {
          // use absolute error bound
          const double scaled = abs_orig_f * inv_eb;
          long long bin = (long long)std::round(scaled);
          const double rnd = inv_mask * (d_QUANT_ABS_REL_R_f64_hash(idx + len) & mask) - 0.5;  // random noise
          double recon = (bin + rnd) * eb;
          recon = max(recon, boundary);
          bin += bnd_bin + 1;
          if ((bin >= maxbin) || (abs_orig_f >= threshold) || (std::abs(abs_orig_f - recon) > abs_errorbound) || (abs_orig_f != abs_orig_f)) {  // last check is to handle NaNs
            assert(((data_i[idx] >> mantissabits) & 0x7ff) != 0);
          } else {
            bin = (bin << 1) ^ (bin >> 63);  // TCMS encoding
            bin = (bin + 1) << 1;
            if (orig_i < 0) bin |= 1;  // include sign
            output = signexpomask | bin;  // 'sign' and 'exponent' fields are all ones, 'mantissa' is non-zero (looks like a negative NaN)
          }
        } else {
          // use relative error bound
          const double log_f = d_QUANT_ABS_REL_R_f64_log2approxf(abs_orig_f);
          const double scaled = log_f * inv_log2eb;
          long long bin = (long long)std::round(scaled);
          const long long rnd1 = d_QUANT_ABS_REL_R_f64_hash(bin + idx + 37);
          const long long rnd2 = d_QUANT_ABS_REL_R_f64_hash((bin >> 32) - idx - 37);
          const double rnd = inv_mask * (((rnd2 << 32) | rnd1) & mask) - 0.5;  // random noise
          double abs_recon_f = d_QUANT_ABS_REL_R_f64_pow2approxf((bin + rnd) * log2eb);
          abs_recon_f = min(abs_recon_f, boundary);
          const double lower = abs_orig_f / (1 + rel_errorbound);
          const double upper = abs_orig_f * (1 + rel_errorbound);
          if (!((bin >= maxbin) || (bin <= -maxbin) || (abs_orig_f >= threshold) || (abs_recon_f < lower) || (abs_recon_f > upper) || (abs_recon_f == 0) || !isfinite(abs_recon_f))) {
            bin = (bin << 1) ^ (bin >> 63);  // TCMS encoding
            bin = (bin + 1) << 1;
            if (orig_i < 0) bin |= 1;  // include sign
            output = signexpomask | bin;  // 'sign' and 'exponent' fields are all ones, 'mantissa' is non-zero (looks like a negative NaN)
          }
        }
      }
    }
    data_i[idx] = (output ^ signexpomask) - 1;
  }
}


static __global__ void d_iQUANT_ABS_REL_R_f64_kernel(const long long len, byte* const __restrict__ data, const double eb, const double log2eb, const double boundary, const double bnd_log_f, const double bnd_scaled, const double bnd_bin)
{
  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const long long mask = (1LL << mantissabits) - 1;
  const double inv_mask = 1.0 / mask;
  const long long signexpomask = ~0LL << mantissabits;
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    const long long val = (data_i[idx] + 1) ^ signexpomask;
    if (((val & signexpomask) == signexpomask) && ((val & ~signexpomask) != 0)) {  // is encoded value
      if (val == (signexpomask | 1)) {
        data_i[idx] = 0;
      } else {
        const long long dec = ((val & ~signexpomask) >> 1) - 1;
        long long bin = (dec >> 1) ^ (((dec << 63) >> 63));  // TCMS decoding
        if (bin > bnd_bin) {
          // use absolute error bound
          bin -= bnd_bin + 1;
          const double rnd = inv_mask * (d_QUANT_ABS_REL_R_f64_hash(idx + len) & mask) - 0.5;  // random noise
          double abs_recon_f = (bin + rnd) * eb;
          abs_recon_f = max(abs_recon_f, boundary);
          data_f[idx] = (val & 1) ? -abs_recon_f : abs_recon_f;
        } else {
          // use relative error bound
          const long long rnd1 = d_QUANT_ABS_REL_R_f64_hash(bin + idx + 37);
          const long long rnd2 = d_QUANT_ABS_REL_R_f64_hash((bin >> 32) - idx - 37);
          const double rnd = inv_mask * (((rnd2 << 32) | rnd1) & mask) - 0.5;  // random noise
          double abs_recon_f = d_QUANT_ABS_REL_R_f64_pow2approxf((bin + rnd) * log2eb);
          abs_recon_f = min(abs_recon_f, boundary);
          data_f[idx] = (val & 1) ? -abs_recon_f : abs_recon_f;
        }
      }
    } else {
      data_i[idx] = val;
    }
  }
}


static inline void d_QUANT_ABS_REL_R_f64(long long& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_ABS_REL_R_f64 ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  const long long len = size / sizeof(double);
  if ((paramc != 2) && (paramc != 3)) {fprintf(stderr, "USAGE: QUANT_ABS_REL_R_f64(absolute_error_bound, relative_error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const double abs_errorbound = paramv[0];
  const double rel_errorbound = paramv[1];
  const double threshold = (paramc == 3) ? paramv[2] : std::numeric_limits<double>::infinity();
  if (abs_errorbound < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_ABS_REL_R_f64: ERROR: absolute_error_bound must be at least %e\n", std::numeric_limits<double>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value
  if (rel_errorbound < 1E-7) {fprintf(stderr, "QUANT_ABS_REL_R_f64 ERROR: relative_error_bound must be at least %e\n", 1E-5f); throw std::runtime_error("LC error");}  // log and exp are too inaccurate below this error bound
  if (threshold <= abs_errorbound) {fprintf(stderr, "QUANT_ABS_REL_R_f64 ERROR: threshold must be larger than absolute_error_bound\n"); throw std::runtime_error("LC error");}
  if (threshold <= rel_errorbound) {fprintf(stderr, "QUANT_ABS_REL_R_f64 ERROR: threshold must be larger than relative_error_bound\n"); throw std::runtime_error("LC error");}

  const double log2eb = d_QUANT_ABS_REL_R_f64_log2approxf(1 + rel_errorbound);
  const double inv_log2eb = 1 / log2eb;
  const double eb = abs_errorbound;
  const double inv_eb = 1 / abs_errorbound;
  const double boundary = abs_errorbound / rel_errorbound;
  const double bnd_log_f = d_QUANT_ABS_REL_R_f64_log2approxf(boundary);
  const double bnd_scaled = bnd_log_f * inv_log2eb;
  const long long bnd_bin = (long long)std::round(bnd_scaled);

  d_QUANT_ABS_REL_R_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, abs_errorbound, rel_errorbound, eb, inv_eb, log2eb, inv_log2eb, threshold, boundary, bnd_log_f, bnd_scaled, bnd_bin);
}


static inline void d_iQUANT_ABS_REL_R_f64(long long& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_ABS_REL_R_f64 ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  const long long len = size / sizeof(double);
  if ((paramc != 2) && (paramc != 3)) {fprintf(stderr, "USAGE: QUANT_ABS_REL_R_f64(absolute_error_bound, relative_error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const double abs_errorbound = paramv[0];
  const double rel_errorbound = paramv[1];
  if (abs_errorbound < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_ABS_REL_R_f64: ERROR: absolute_error_bound must be at least %e\n", std::numeric_limits<double>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value
  if (rel_errorbound < 1E-7) {fprintf(stderr, "QUANT_ABS_REL_R_f64 ERROR: relative_error_bound must be at least %e\n", 1E-5f); throw std::runtime_error("LC error");}  // log and exp are too inaccurate below this error bound

  const double eb = abs_errorbound;
  const double log2eb = d_QUANT_ABS_REL_R_f64_log2approxf(1 + rel_errorbound);
  const double inv_log2eb = 1.0 / log2eb;
  const double boundary = abs_errorbound / rel_errorbound;
  const double bnd_log_f = d_QUANT_ABS_REL_R_f64_log2approxf(boundary);
  const double bnd_scaled = bnd_log_f * inv_log2eb;
  const long long bnd_bin = (long long)std::round(bnd_scaled);

  d_iQUANT_ABS_REL_R_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, eb, log2eb, boundary, bnd_log_f, bnd_scaled, bnd_bin);
}
