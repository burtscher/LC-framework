/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
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


#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


static __global__ void d_QUANT_NOA_0_f64_kernel(const long long len, byte* const __restrict__ data, byte* const __restrict__ orig_data, const double errorbound, const double* maxf, const double* minf, const double threshold)
{
  double* const orig_data_f = (double*)orig_data;
  long long* const orig_data_i = (long long*)orig_data;
  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data;

  const double adj_eb = (*maxf - *minf) * errorbound;
  if (adj_eb < cuda::std::numeric_limits<double>::min()) {printf("QUANT_NOA_0_f64: ERROR: error_bound must be at least %e, NOA error bound was calculated to be %e\n", cuda::std::numeric_limits<double>::min(), adj_eb); __trap();}
  const double eb2 = 2 * adj_eb;
  const double inv_eb2 = 0.5 / adj_eb;
  const int mantissabits = 52;
  const long long maxbin = 1LL << (mantissabits - 1);  // leave 1 bit for sign
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    const double orig_f = orig_data_f[idx];
    const double scaled = orig_f * inv_eb2;
    const long long bin = (long long)std::round(scaled);
    const double recon = bin * eb2;

    if ((bin >= maxbin) || (bin <= -maxbin) || (std::abs(orig_f) >= threshold) || (std::abs(orig_f - recon) > adj_eb) || (orig_f != orig_f)) {  // last check is to handle NaNs
      data_f[idx] = orig_f;
      assert(((orig_data_i[idx] >> mantissabits) & 0x7ff) != 0);
    } else {
      data_i[idx] = (bin << 1) ^ (bin >> 63);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
  }

  if (idx == 0) {
    data_f[len] = adj_eb;
  }
}


static __global__ void d_iQUANT_NOA_0_f64_kernel(const long long len, byte* const __restrict__ data)
{
  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data;

  const double eb2 = 2 * data_f[len];
  const int mantissabits = 52;
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    long long bin = data_i[idx];
    if ((0 <= bin) && (bin < (1LL << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 63) >> 63));  // TCMS decoding
      data_f[idx] = bin * eb2;
    }
  }
}


static inline void d_QUANT_NOA_0_f64(long long& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  const long long len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_NOA_0_f64(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}
  const double errorbound = paramv[0];
  const double threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<double>::infinity();
  if (errorbound < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: error_bound must be at least %e\n", std::numeric_limits<double>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value
  if (threshold <= errorbound) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: threshold must be larger than error_bound\n"); throw std::runtime_error("LC error");}

  byte* d_new_data;
  if (cudaSuccess != cudaMalloc((void**) &d_new_data, size + sizeof(double))) {
    fprintf(stderr, "ERROR: could not allocate d_new_data\n\n");
    throw std::runtime_error("LC error");
  }

  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast((double*)data);
  thrust::pair<thrust::device_ptr<double>, thrust::device_ptr<double>> min_max = thrust::minmax_element(thrust::device, dev_ptr, dev_ptr + len);

  d_QUANT_NOA_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, d_new_data, data, errorbound, thrust::raw_pointer_cast(min_max.second), thrust::raw_pointer_cast(min_max.first), threshold);

  cudaFree(data);
  data = (byte*) d_new_data;
  size += sizeof(double);
}


static inline void d_iQUANT_NOA_0_f64(long long& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_NOA_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  const long long len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {fprintf(stderr, "USAGE: QUANT_NOA_0_f64(error_bound [, threshold])\n"); throw std::runtime_error("LC error");}

  d_iQUANT_NOA_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len - 1, data);

  size -= sizeof(double);
}
