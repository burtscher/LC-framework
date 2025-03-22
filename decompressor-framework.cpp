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


static void h_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize)
{
  // input header
  long long* const head_in = (long long*)input;
  outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];
  long long* const start = new long long [chunks];

  // convert chunk sizes into starting positions
  long long pfs = 0;
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (long long)size_in[chunkID];
  }

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      // decompress
      memcpy(out, &data_in[start[chunkID]], csize);

      // decode
      /*##comp-decoder-beg##*/
      std::swap(in, out);
      h_iCLOG_2(csize, in, out);
      /*##comp-decoder-end##*/

      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %lld\n\n", csize, osize, chunkID); throw std::runtime_error("LC error");}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}


int main(int argc, char* argv [])
{
  /*##print-beg##*/
  /*##print-end##*/
  printf("Copyright 2024 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s compressed_file_name decompressed_file_name [performance_analysis (y)]\n\n", argv[0]); return -1;}

  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  long long pre_size = 0;
  const long long pre_val = fread(&pre_size, sizeof(pre_size), 1, fin); assert(pre_val == sizeof(pre_size));
  fseek(fin, 0, SEEK_END);
  const long long hencsize = ftell(fin);  assert(hencsize > 0);
  byte* const hencoded = new byte [std::max(pre_size, hencsize)];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(hencoded, 1, hencsize, fin);  assert(insize == hencsize);
  fclose(fin);
  printf("encoded size: %lld bytes\n", insize);

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
  byte* hdecoded = new byte [pre_size];
  long long hdecsize = 0;

  if (perf) {
    // warm up
    byte* dummy = new byte [pre_size];
    long long dummy_size = 0;
    h_decode(hencoded, dummy, dummy_size);
    delete [] dummy;
  }

  // time CPU decoding
  CPUTimer htimer;
  htimer.start();
  h_decode(hencoded, hdecoded, hdecsize);
  /*##pre-decoder-beg##*/
  /*##pre-decoder-end##*/
  double hruntime = htimer.stop();

  printf("decoded size: %lld bytes\n", hdecsize);
  const float CR = (100.0 * insize) / hdecsize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);
  if (perf) {
    printf("decoding time: %.6f s\n", hruntime);
    double hthroughput = hdecsize * 0.000000001 / hruntime;
    printf("decoding throughput: %8.3f Gbytes/s\n", hthroughput);
  }

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hdecoded, 1, hdecsize, fout);
  fclose(fout);

  delete [] hencoded;
  delete [] hdecoded;
  return 0;
}
