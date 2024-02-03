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


// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ unsigned int d_QUANT_REL_R_f32_hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


static __host__ __device__ inline float d_QUANT_REL_R_f32_log2approxf(const float orig_f)
{
  const int mantissabits = 23;
  const int orig_i = *((int*)&orig_f);
  const int expo = (orig_i >> mantissabits) & 0xff;
  const int frac_i = (127 << mantissabits) | (orig_i & ~(~0 << mantissabits));
  const float frac_f = *((float*)&frac_i);
  const float log_f = frac_f + (expo - 128);  // - bias - 1
  return log_f;
}


static __device__ inline float d_QUANT_REL_R_f32_pow2approxf(const float log_f)
{
  const int mantissabits = 23;
  const float biased = log_f + 127;
  const int expo = biased;
  const float frac_f = biased - (expo - 1);
  const int frac_i = *((int*)&frac_f);
  const int exp_i = (expo << mantissabits) | (frac_i & ~(~0 << mantissabits));
  const float recon_f = *((float*)&exp_i);
  return recon_f;
}


static __global__ void d_QUANT_REL_R_f32_kernel(const int len, byte* const __restrict__ data, const float errorbound, const float log2eb, const float inv_log2eb, const float threshold)
{
  int* const data_i = (int*)data;

  const int mantissabits = 23;
  const int signexpomask = ~0 << mantissabits;
  const int maxbin = (1 << (mantissabits - 2)) - 1;  // leave 2 bits for 2 signs (plus one element)
  const int mask = (1 << mantissabits) - 1;
  const float inv_mask = 1.0f / mask;

  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    const int orig_i = data_i[idx];
    const int abs_orig_i = orig_i & 0x7fff'ffff;
    const float abs_orig_f = *((float*)&abs_orig_i);
    int output = orig_i;
    const int expo = (orig_i >> mantissabits) & 0xff;
    if (expo == 0) {  // zero or de-normal values
      if (abs_orig_i == 0) {  // zero
        output = signexpomask | 1;
      }
    } else {
      if (expo == 0xff) {  // INF or NaN
        if (((orig_i & signexpomask) == signexpomask) && ((orig_i & ~signexpomask) != 0)) {  // negative NaN
          output = abs_orig_i;  // make positive NaN
        }
      } else {  // normal value
        const float log_f = d_QUANT_REL_R_f32_log2approxf(abs_orig_f);
        const float scaled = log_f * inv_log2eb;
        int bin = (int)roundf(scaled);
        const float rnd = inv_mask * (d_QUANT_REL_R_f32_hash(idx + 17 + len) & mask) - 0.5f;  // random noise
        const float abs_recon_f = d_QUANT_REL_R_f32_pow2approxf((bin + rnd) * log2eb);
        const float lower = abs_orig_f / (1 + errorbound);
        const float upper = abs_orig_f * (1 + errorbound);
        if (!((bin >= maxbin) || (bin <= -maxbin) || (abs_orig_f >= threshold) || (abs_recon_f < lower) || (abs_recon_f > upper) || (abs_recon_f == 0) || !isfinite(abs_recon_f))) {
          bin = (bin << 1) ^ (bin >> 31);  // TCMS encoding
          bin = (bin + 1) << 1;
          if (orig_i < 0) bin |= 1;  // include sign
          output = signexpomask | bin;  // 'sign' and 'exponent' fields are all ones, 'mantissa' is non-zero (looks like a negative NaN)
        }
      }
    }
    data_i[idx] = (output ^ signexpomask) - 1;
  }
}


static __global__ void d_iQUANT_REL_R_f32_kernel(const int len, byte* const __restrict__ data, const float errorbound, const float log2eb)
{
  float* const data_f = (float*)data;
  int* const data_i = (int*)data;

  const int mantissabits = 23;
  const int signexpomask = ~0 << mantissabits;
  const int mask = (1 << mantissabits) - 1;
  const float inv_mask = 1.0f / mask;

  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    const int val = (data_i[idx] + 1) ^ signexpomask;
    if (((val & signexpomask) == signexpomask) && ((val & ~signexpomask) != 0)) {  // is encoded value
      if (val == (signexpomask | 1)) {
        data_i[idx] = 0;
      } else {
        const int dec = ((val & ~signexpomask) >> 1) - 1;
        const int bin = (dec >> 1) ^ (((dec << 31) >> 31));  // TCMS decoding
        const float rnd = inv_mask * (d_QUANT_REL_R_f32_hash(idx + 17 + len) & mask) - 0.5f;  // random noise
        const float abs_recon_f = d_QUANT_REL_R_f32_pow2approxf((bin + rnd) * log2eb);
        data_f[idx] = (val & 1) ? -abs_recon_f : abs_recon_f;
      }
    } else {
      data_i[idx] = val;
    }
  }
}


static inline void d_QUANT_REL_R_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {fprintf(stderr, "QUANT_REL_R_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); exit(-1);}
  const int len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_REL_R_f32(error_bound [, threshold])\n"); exit(-1);}
  const float errorbound = paramv[0];
  const float threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<float>::infinity();
  if (errorbound < 1E-5f) {fprintf(stderr, "QUANT_REL_R_f32: ERROR: error_bound must be at least %e\n", 1E-5f); exit(-1);}  // minimum positive normalized value
  if (threshold <= errorbound) {fprintf(stderr, "QUANT_REL_R_f32: ERROR: threshold must be larger than error_bound\n"); exit(-1);}

  const float log2eb = d_QUANT_REL_R_f32_log2approxf(1 + errorbound);
  const float inv_log2eb = 1 / log2eb;

  d_QUANT_REL_R_f32_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, errorbound, log2eb, inv_log2eb, threshold);
}


static inline void d_iQUANT_REL_R_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {fprintf(stderr, "QUANT_REL_R_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); exit(-1);}
  const int len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_REL_R_f32(error_bound [, threshold])\n"); exit(-1);}
  const float errorbound = paramv[0];
  if (errorbound < 1E-5f) {fprintf(stderr, "QUANT_REL_R_f32: ERROR: error_bound must be at least %e\n", 1E-5f); exit(-1);}  // minimum positive normalized value

  const float log2eb = d_QUANT_REL_R_f32_log2approxf(1 + errorbound);

  d_iQUANT_REL_R_f32_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data, errorbound, log2eb);
}
