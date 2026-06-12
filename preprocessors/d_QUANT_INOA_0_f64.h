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


static __global__ void d_QUANT_INOA_0_f64_kernel(const long long len, const unsigned long long* const __restrict__ orig_data_u, unsigned long long* const __restrict__ data_u, const int eb_e, const int thr_e, const long long offs)
{
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    const unsigned long long val = orig_data_u[idx];
    const long long abs = val & ((1ULL << (e + m)) - 1ULL);  // compute absolute value
    const int val_e = abs >> m;  // extract exponent
    long long enc = 0LL;  // default value is 0
    if (val_e >= thr_e) {  // at or above threshold
      enc = abs - offs;  // lossless encoding
    } else if (val_e >= eb_e) {  // lossy encoding
      long long mant = val & ((1LL << m) - 1LL);  // extract mantissa
      const int shift = thr_e - val_e;  // bias cancels out
      mant |= 1LL << m;  // insert implicit 1
      mant += 1LL << (shift - 1);  // round to nearest, ties round away from zero
      enc = mant >> shift;  // shift out unnecessary bits
    }
    enc = (enc << 1) | (~val >> (e + m));  // magnitude ~sign
    if (enc != 0LL) enc--;  // -0 -> +0 and fill gap
    data_u[idx] = enc;
  }
}


static __global__ void d_iQUANT_INOA_0_f64_kernel(const long long len, unsigned long long* const __restrict__ data_u, const int thr_e, const long long offs)
{
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    const unsigned long long enc = data_u[idx];
    long long dec = 0LL;  // default value is 0
    if (enc != 0LL) {
      const long long abs = (enc + 1) >> 1;  // absolute value
      if (abs >= (1LL << m)) {  // above threshold
        dec = abs + offs;  // decode losslessly
      } else {  // non-zero lossy case
        const int shift = __builtin_clzll(abs) - (63 - m);  // compute shift amount
        dec = abs << shift;  // shift to normalized position
        dec &= (1LL << m) - 1LL;  // remove implied 1
        dec |= (long long)(thr_e - shift) << m;  // insert biased exponent
      }
      dec |= enc << (e + m);  // insert sign bit
    }
    data_u[idx] = dec;
  }
}


static inline void d_QUANT_INOA_0_f64(long long& size, byte*& data, const int paramc, const double paramv [])
{
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  if (size % sizeof(double) != 0) {fprintf(stderr, "QUANT_INOA_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  if (paramc != 1) {fprintf(stderr, "USAGE: QUANT_INOA_0_f64(error_bound)\n"); throw std::runtime_error("LC error");}

  const long long len = size / sizeof(double);
  double* const orig_data_f = (double*)data;
  const unsigned long long* const orig_data_u = (unsigned long long*)data;
  unsigned long long* data_u;
  if (cudaSuccess != cudaMalloc((void**)&data_u, size + sizeof(unsigned long long))) {fprintf(stderr, "ERROR: could not allocate data_u on device\n"); throw std::runtime_error("LC error");}

  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(orig_data_f);
  thrust::pair<thrust::device_ptr<double>, thrust::device_ptr<double>> min_max = thrust::minmax_element(thrust::device, dev_ptr, dev_ptr + len);
  double minf, maxf;
  cudaMemcpy(&minf, thrust::raw_pointer_cast(min_max.first), sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&maxf, thrust::raw_pointer_cast(min_max.second), sizeof(double), cudaMemcpyDeviceToHost);

  const double errorbound = (maxf - minf) * paramv[0];  // normalize
  if (errorbound < std::numeric_limits<double>::min()) {fprintf(stderr, "QUANT_INOA_0_f64: ERROR: error_bound must be at least %e\n", std::numeric_limits<double>::min()); throw std::runtime_error("LC error");}  // minimum positive normalized value

  const int eb_e = (*((long long*)&errorbound) >> m) & ((1LL << e) - 1LL);  // extract biased exponent
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  const long long offs = ((long long)thr_e << m) - (1LL << m);  // offset for lossless encoding
  if (thr_e >= (1 << e) - 1) {fprintf(stderr, "QUANT_INOA_0_f64: ERROR: error_bound is too large\n"); throw std::runtime_error("LC error");}

  const long long temp = eb_e;
  cudaMemcpyAsync(&data_u[len], &temp, sizeof(long long), cudaMemcpyHostToDevice);
  d_QUANT_INOA_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, orig_data_u, data_u, eb_e, thr_e, offs);

  cudaFree(data);
  data = (byte*)data_u;
  size += sizeof(double);
}


static inline void d_iQUANT_INOA_0_f64(long long& size, byte*& data, const int paramc, const double paramv [])
{
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  if (size % sizeof(double) != 0) {fprintf(stderr, "iQUANT_INOA_0_f64: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); throw std::runtime_error("LC error");}
  if (paramc != 1) {fprintf(stderr, "USAGE: iQUANT_INOA_0_f64(error_bound)\n"); throw std::runtime_error("LC error");}

  size -= sizeof(double);
  const long long len = size / sizeof(double);
  unsigned long long* const data_u = (unsigned long long*)data;

  long long temp;
  cudaMemcpy(&temp, &data_u[len], sizeof(long long), cudaMemcpyDeviceToHost);  // use exponent of adjusted error bound (ignore passed parameter)
  const int eb_e = temp;
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  const long long offs = ((long long)thr_e << m) - (1LL << m);  // offset for lossless encoding
  if (thr_e >= (1 << e) - 1) {fprintf(stderr, "iQUANT_INOA_0_f64: ERROR: error_bound is too large\n"); throw std::runtime_error("LC error");}

  d_iQUANT_INOA_0_f64_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, (unsigned long long*)data, thr_e, offs);
}
