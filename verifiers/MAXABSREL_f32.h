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


static void MAXABSREL_f32(const long long size, const byte* const __restrict__ recon, const byte* const __restrict__ orig, const int paramc, const double paramv [])
{
  using type_f = float;
  using type_i = int;
  assert(sizeof(type_f) == sizeof(type_i));

  if ((size % sizeof(type_f)) != 0) {fprintf(stderr, "ERROR: MAXABSREL_f32 requires data to be a multiple of %ld bytes long\n", sizeof(type_f)); throw std::runtime_error("LC error");}
  if (paramc != 2) {fprintf(stderr, "ERROR: MAXABSREL_f32 requires two parameters that specify the maximum allowed absolute and relative errors\n"); throw std::runtime_error("LC error");}
  const type_f abs_errorbound = paramv[0];
  const type_f rel_errorbound = paramv[1];
  if (abs_errorbound <= 0) {fprintf(stderr, "ERROR: MAXABSREL_f32 requires the maximum allowed absolute error to be greater than zero\n"); throw std::runtime_error("LC error");}
  if (rel_errorbound <= 0) {fprintf(stderr, "ERROR: MAXABSREL_f32 requires the maximum allowed relative error to be greater than zero\n"); throw std::runtime_error("LC error");}

  const type_f* const orig_f = (type_f*)orig;
  const type_f* const recon_f = (type_f*)recon;
  const long long len = size / sizeof(type_f);

  for (long long i = 0; i < len; i++) {
    if (!std::isfinite(orig_f[i]) || !std::isfinite(recon_f[i])) {  // at least one value is INF or NaN
      if (recon_f[i] != orig_f[i]) {
        if (!std::isnan(orig_f[i]) || !std::isnan(recon_f[i])) {  // at least one value isn't a NaN
          fprintf(stderr, "MAXABSREL_f32 ERROR: absolute error bound exceeded due to NaN or INF at position %lld: value is '%.10f' vs '%.10f'\n\n", i, recon_f[i], orig_f[i]);
          throw std::runtime_error("LC error");
        }
      }
    } else {
      // ABS
      type_f lower = orig_f[i] - abs_errorbound;
      type_f upper = orig_f[i] + abs_errorbound;
      if ((recon_f[i] < lower) || (recon_f[i] > upper) || (std::abs(orig_f[i] - recon_f[i]) > abs_errorbound)) {
        fprintf(stderr, "MAXABSREL_f32 ERROR: absolute error bound of %.10f exceeded at position %lld: value is '%.10f' vs '%.10f' (diff is '%.10f')\n\n", abs_errorbound, i, recon_f[i], orig_f[i], std::abs(orig_f[i] - recon_f[i]));
        throw std::runtime_error("LC error");
      }
      // REL
      const type_f abs_orig_f = std::abs(orig_f[i]);
      const type_f abs_recon_f = std::abs(recon_f[i]);
      lower = abs_orig_f / (1 + rel_errorbound);
      upper = abs_orig_f * (1 + rel_errorbound);
      if ((std::signbit(orig_f[i]) != std::signbit(recon_f[i])) || (abs_recon_f < lower) || (abs_recon_f > upper)) {
        fprintf(stderr, "MAXABSREL_f32 ERROR: relative error bound of %e exceeded at position %lld: value is '%e' vs '%e'\n\n", rel_errorbound, i, recon_f[i], orig_f[i]);
        throw std::runtime_error("LC error");
      }
    }
  }

  printf("MAXABSREL_f32 verification passed\n");
}
