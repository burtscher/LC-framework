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


static void MAXR2R_f64(const int size, const byte* const __restrict__ recon, const byte* const __restrict__ orig, const int paramc, const double paramv [])
{
  using type_f = double;
  using type_i = long long;
  assert(sizeof(type_f) == sizeof(type_i));

  if ((size % sizeof(type_f)) != 0) {fprintf(stderr, "ERROR: MAXR2R_f64 requires data to be a multiple of %ld bytes long\n", sizeof(type_f)); throw std::runtime_error("LC error");}
  if (paramc != 1) {fprintf(stderr, "ERROR: MAXR2R_f64 requires one parameter that specifies the maximum allowed relative-to-range error\n"); throw std::runtime_error("LC error");}
  const type_f errorbound = paramv[0];
  if (errorbound <= 0) {fprintf(stderr, "ERROR: MAXR2R_f64 requires the maximum allowed relative-to-range error to be greater than zero\n"); throw std::runtime_error("LC error");}

  const type_f* const orig_f = (type_f*)orig;
  const type_f* const recon_f = (type_f*)recon;
  const int len = size / sizeof(type_f);
  type_f maxf, minf;
  maxf = minf = orig_f[0];
  #pragma omp parallel for default(none) shared(orig_f, len) reduction(max:maxf) reduction(min:minf)
  for (int i = 1; i < len; i++) {
    if (std::isfinite(orig_f[i]) && !std::isnan(orig_f[i])) {
      maxf = (orig_f[i] > maxf) ? orig_f[i] : maxf;
      minf = (orig_f[i] < minf) ? orig_f[i] : minf;
    }
  }

  const type_f adj_eb = (maxf - minf) * errorbound;
  for (int i = 0; i < len; i++) {
    if (!std::isfinite(orig_f[i]) || !std::isfinite(recon_f[i])) {  // at least one value is INF or NaN
      if (recon_f[i] != orig_f[i]) {
        if (!std::isnan(orig_f[i]) || !std::isnan(recon_f[i])) {  // at least one value isn't a NaN
          fprintf(stderr, "MAXR2R_f64 ERROR: adjusted error bound exceeded due to NaN or INF at position %d: value is '%.10f' vs '%.10f'\n\n", i, recon_f[i], orig_f[i]);
          throw std::runtime_error("LC error");
        }
      }
    } else {
      const type_f lower = orig_f[i] - adj_eb;
      const type_f upper = orig_f[i] + adj_eb;
      if ((recon_f[i] < lower) || (recon_f[i] > upper) || (fabsf(orig_f[i] - recon_f[i]) > adj_eb)) {
        fprintf(stderr, "MAXR2R_f64 ERROR: adjusted error bound of %.10f exceeded at position %d: value is '%.10f' vs '%.10f' (diff is '%.10f')\n\n", adj_eb, i, recon_f[i], orig_f[i], fabsf(orig_f[i] - recon_f[i]));
        throw std::runtime_error("LC error");
      }
    }
  }

  printf("MAXR2R_f64 verification passed\n");
}
