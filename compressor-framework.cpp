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


#define NDEBUG

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#define WS 64
#else
#define WS 32
#endif

#include <limits>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <sys/time.h>
#include "include/macros.h"
/*##include-beg##*/
/*##include-end##*/


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static void h_encode(const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long& outsize)
{
  // initialize
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  long long* const carry = new long long [chunks];
  memset(carry, 0, chunks * sizeof(long long));

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, insize - base);
    memcpy(out, &input[base], osize);

    // encode chunk
    int csize = osize;
    bool good = true;
    /*##comp-encoder-beg##*/
    if (good) {
      std::swap(in, out);
      good = h_CLOG_2(csize, in, out);
    }
    /*##comp-encoder-end##*/

    // handle carry and store chunk
    long long offs = 0LL;
    if (chunkID > 0) {
      do {
        #pragma omp atomic read
        offs = carry[chunkID - 1];
      } while (offs == 0);
      #pragma omp flush
    }
    if (good && (csize < osize)) {
      // store compressed data
      #pragma omp atomic write
      carry[chunkID] = (offs + (long long)csize);
      size_out[chunkID] = csize;
      memcpy(&data_out[offs], out, csize);
    } else {
      // store original data
      #pragma omp atomic write
      carry[chunkID] = (offs + (long long)osize);
      size_out[chunkID] = osize;
      memcpy(&data_out[offs], &input[base], osize);
    }
  }

  // output header
  head_out[0] = insize;

  // finish
  outsize = &data_out[carry[chunks - 1]] - output;
  delete [] carry;
}


int main(int argc, char* argv [])
{
  /*##print-beg##*/
  /*##print-end##*/
  printf("Copyright 2024 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s input_file_name compressed_file_name [performance_analysis(y)]\n\n", argv[0]); return -1;}
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); throw std::runtime_error("LC error");}
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %lld bytes\n", insize);

  // Check if the third argument is "y" to enable performance analysis
  char* perf_str = argv[3];
  bool perf = false;
  if (perf_str != nullptr && strcmp(perf_str, "y") == 0) {
    perf = true;
  } else if (perf_str != nullptr && strcmp(perf_str, "y") != 0) {
    fprintf(stderr, "ERROR: Invalid argument. Use 'y' or nothing.\n");
    throw std::runtime_error("LC error");
  }

  // allocate CPU memory
  const long long chunks = (insize + CS - 1) / CS;  // round up
  const long long maxsize = 2 * sizeof(long long) + chunks * sizeof(short) + chunks * CS;
  byte* const hencoded = new byte [maxsize];
  long long hencsize = 0;

  // time CPU preprocessor encoding
  byte* hpreencdata = new byte [insize];
  std::copy(input, input + insize, hpreencdata);
  long long hpreencsize = insize;
  if (perf) {
    // warm up
    byte* dummy = new byte [insize];
    std::copy(input, input + insize, dummy);
    long long dummy_size = 0;
    h_encode(dummy, dummy_size, hencoded, hencsize);
    delete [] dummy;
  }

  CPUTimer htimer;
  htimer.start();
  /*##pre-encoder-beg##*/
  /*##pre-encoder-end##*/
  h_encode(hpreencdata, hpreencsize, hencoded, hencsize);
  double hruntime = htimer.stop();

  printf("encoded size: %lld bytes\n", hencsize);
  const float CR = (100.0 * hencsize) / insize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);
  if (perf) {
    printf("encoding time: %.6f s\n", hruntime);
    double hthroughput = insize * 0.000000001 / hruntime;
    printf("encoding throughput: %8.3f Gbytes/s\n", hthroughput);
  }

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hencoded, 1, hencsize, fout);
  fclose(fout);

  delete [] input;
  delete [] hencoded;
  return 0;
}
