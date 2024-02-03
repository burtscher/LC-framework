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


static inline void h_QUANT_R2R_0_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {fprintf(stderr, "QUANT_R2R_0_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); exit(-1);}
  const int len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_R2R_0_f32(error_bound [, threshold])\n"); exit(-1);}
  const float errorbound = paramv[0];
  const float threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<float>::infinity();

  float* const orig_data_f = (float*)data;
  float* const data_f = new float [len + 1];
  int* const data_i = (int*)data_f;

  float maxf, minf;
  maxf = minf = orig_data_f[0];
  #pragma omp parallel for default(none) shared(len, orig_data_f) reduction(max:maxf) reduction(min:minf)
  for (int i = 0; i < len; i++) {
    const float orig_val = orig_data_f[i];
    if (std::isfinite(orig_val)) {
      maxf = std::max(orig_val, maxf);
      minf = std::min(orig_val, minf);
    }
  }

  const float adj_eb = (maxf - minf) * errorbound;
  data_f[len] = adj_eb;
  if (adj_eb < std::numeric_limits<float>::min()) {fprintf(stderr, "QUANT_R2R_0_f32: ERROR: error_bound must be at least %e, R2R error bound was calculated to be %e\n", std::numeric_limits<float>::min(), adj_eb); exit(-1);}  // minimum positive normalized value
  if (threshold <= adj_eb) {fprintf(stderr, "QUANT_R2R_0_f32: ERROR: threshold must be larger than error_bound, R2R error bound was calculated to be %e\n", adj_eb); exit(-1);}

  const int mantissabits = 23;
  const int maxbin = 1 << (mantissabits - 1);  // leave 1 bit for sign
  const float eb2 = 2 * adj_eb;
  const float inv_eb2 = 0.5f / adj_eb;

  int count = 0;
  #pragma omp parallel for default(none) shared(len, data, data_i, data_f, orig_data_f, adj_eb, eb2, inv_eb2, threshold, maxbin, mantissabits) reduction(+: count)
  for (int i = 0; i < len; i++) {
    const float orig_f = orig_data_f[i];
    const float scaled = orig_f * inv_eb2;
    const int bin = (int)roundf(scaled);
    const float recon = bin * eb2;

    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(orig_f) >= threshold) || (recon < orig_f - adj_eb) || (recon > orig_f + adj_eb) || (fabsf(orig_f - recon) > adj_eb) || (orig_f != orig_f)) {  // last check is to handle NaNs
      count++;  // informal only
      assert(((((int*)data)[i] >> mantissabits) & 0xff) != 0);
      data_f[i] = orig_f;
    } else {
      data_i[i] = (bin << 1) ^ (bin >> 31);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
  }

  delete [] data;
  data = (byte *)data_f;
  size += sizeof(float);

  if (count != 0) printf("QUANT_R2R_0_f32: encountered %d non-quantizable values (%.3f%%)\n", count, 100.0 * count / len);  // informal only
}


static inline void h_iQUANT_R2R_0_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {fprintf(stderr, "QUANT_R2R_0_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); exit(-1);}
  const int len = size / sizeof(float) - 1;
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_R2R_0_f32(error_bound [, threshold])\n"); exit(-1);}

  float* const data_f = (float*)data;
  int* const data_i = (int*)data_f;
  const float errorbound = data_f[len];
  if (errorbound < std::numeric_limits<float>::min()) {fprintf(stderr, "QUANT_R2R_0_f32: ERROR: error_bound must be at least %e\n", std::numeric_limits<float>::min()); exit(-1);}  // minimum positive normalized value

  const int mantissabits = 23;
  const float eb2 = 2 * errorbound;

  #pragma omp parallel for default(none) shared(len, data_f, data_i, eb2, mantissabits)
  for (int i = 0; i < len; i++) {
    int bin = data_i[i];
    if ((0 <= bin) && (bin < (1 << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 31) >> 31));  // TCMS decoding
      data_f[i] = bin * eb2;
    }
  }

  size -= sizeof(float);
}
