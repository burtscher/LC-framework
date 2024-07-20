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


#include <limits>


static inline void h_QUANT_NOA_0_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); throw std::runtime_error("LC error");}
  const int len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_NOA_0_f64(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const double errorbound = paramv[0];
  const double threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<double>::infinity();

  double* const orig_data_f = (double*)data;
  double* const data_f = new double [len + 1];
  long long* const data_i = (long long*)data_f;

  double maxf, minf;
  maxf = minf = orig_data_f[0];
  #pragma omp parallel for default(none) shared(len, orig_data_f) reduction(max:maxf) reduction(min:minf)
  for (int i = 0; i < len; i++) {
    const double orig_val = orig_data_f[i];
    if (std::isfinite(orig_val)) {
      maxf = std::max(orig_val, maxf);
      minf = std::min(orig_val, minf);
    }
  }

  const double adj_eb = (maxf - minf) * errorbound;
  data_f[len] = adj_eb;
  if (adj_eb < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: error_bound must be at least %e, NOA error bound was calculated to be %e\n", std::numeric_limits<double>::min(), adj_eb); throw std::runtime_error("LC error");}  // minimum positive normalized value
  if (threshold <= adj_eb) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: threshold must be larger than error_bound, NOA error bound was calculated to be %e\n", adj_eb); throw std::runtime_error("LC error");}

  const int mantissabits = 52;
  const long long maxbin = 1LL << (mantissabits - 1);  // leave 1 bit for sign
  const double eb2 = 2 * adj_eb;
  const double inv_eb2 = 0.5 / adj_eb;

  int count = 0;
  #pragma omp parallel for default(none) shared(len, data, data_i, data_f, orig_data_f, adj_eb, eb2, inv_eb2, threshold, maxbin, mantissabits) reduction(+: count)
  for (int i = 0; i < len; i++) {
    const double orig_f = orig_data_f[i];
    const double scaled = orig_f * inv_eb2;
    const long long bin = (long long)round(scaled);
    const double recon = bin * eb2;

    if ((bin >= maxbin) || (bin <= -maxbin) || (fabs(orig_f) >= threshold) || (fabs(orig_f - recon) > adj_eb) || (orig_f != orig_f)) {  // last check is to handle NaNs
      count++;  // informal only
      assert(((((long long*)data)[i] >> mantissabits) & 0x7ff) != 0);
      data_f[i] = orig_f;
    } else {
      data_i[i] = (bin << 1) ^ (bin >> 63);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
  }

  delete [] data;
  data = (byte *)data_f;
  size += sizeof(double);

  if (count != 0) printf("QUANT_NOA_0_f64: encountered %d non-quantizable values (%.3f%%)\n", count, 100.0 * count / len);  // informal only
}


static inline void h_iQUANT_NOA_0_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); throw std::runtime_error("LC error");}
  const int len = size / sizeof(double) - 1;
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_NOA_0_f64(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}

  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data_f;
  const double errorbound = data_f[len];
  if (errorbound < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: error_bound must be at least %e\n", std::numeric_limits<float>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value

  const int mantissabits = 52;
  const double eb2 = 2 * errorbound;

  #pragma omp parallel for default(none) shared(len, data_f, data_i, eb2, mantissabits)
  for (int i = 0; i < len; i++) {
    long long bin = data_i[i];
    if ((0 <= bin) && (bin < (1LL << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 63) >> 63));  // TCMS decoding
      data_f[i] = bin * eb2;
    }
  }

  size -= sizeof(double);
}