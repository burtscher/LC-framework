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


static void PSNR_f64(const long long size, const byte* const __restrict__ recon, const byte* const __restrict__ orig, const int paramc, const double paramv [])
{
  using type = double;

  if ((size % sizeof(type)) != 0) {fprintf(stderr, "ERROR: PSNR_f64 requires data to be a multiple of %ld bytes long\n", sizeof(type)); throw std::runtime_error("LC error");}
  if (paramc != 1) {fprintf(stderr, "ERROR: PSNR_f64 requires one parameter that specifies the minimum allowed peak signal to noise ratio\n"); throw std::runtime_error("LC error");}

  const type* const orig_t = (type*)orig;
  const type* const recon_t = (type*)recon;
  const long long len = size / sizeof(type);
  const type errorbound = paramv[0];

  double mse = 0;
  type inmin = orig_t[0];
  type inmax = orig_t[0];
  for (long long i = 0; i < len; i++) {
    const type diff = recon_t[i] - orig_t[i];
    mse += diff * diff;
    inmin = std::min(inmin, orig_t[i]);
    inmax = std::max(inmax, orig_t[i]);
  }
  mse /= len;

  if (mse != 0) {
    type psnr = (inmax == inmin) ? -std::numeric_limits<type>::infinity() : (20 * log10(inmax - inmin) - 10 * log10(mse));
    if (psnr < errorbound) {fprintf(stderr, "PSNR_f64 ERROR: decoded data's PSNR of %f does not meet minimum allowed value of %f\n\n", psnr, errorbound); throw std::runtime_error("LC error");}
  }

  printf("PSNR_f64 verification passed\n");
}
