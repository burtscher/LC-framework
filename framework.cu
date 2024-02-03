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


//#define NDEBUG

using byte = unsigned char;

static const int max_stages = 8;  // cannot be more than 8

#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <strings.h>
#include <cassert>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include <regex>
#include <sys/time.h>

#include "include/consts.h"
#ifndef USE_GPU
  #ifndef USE_CPU
  //no CPU and no GPU
  #else
  #include "preprocessors/include/CPUpreprocessors.h"
  #include "components/include/CPUcomponents.h"
  #endif
#else
  #include <cuda.h>
  #include "include/max_reduction.h"
  #include "include/max_scan.h"
  #include "include/prefix_sum.h"
  #include "include/sum_reduction.h"

  #ifndef USE_CPU
  #include "preprocessors/include/GPUpreprocessors.h"
  #include "components/include/GPUcomponents.h"
  #else
  #include "preprocessors/include/preprocessors.h"
  #include "components/include/components.h"
  #endif
#endif
#include "verifiers/include/verifiers.h"


static void verify(const int size, const byte* const recon, const byte* const orig, std::vector<std::pair<byte, std::vector<double>>> verifs)
{
  for (int i = 0; i < verifs.size(); i++) {
    std::vector<double> params = verifs[i].second;
    switch (verifs[i].first) {
      default: fprintf(stderr, "ERROR: unknown verifier\n\n"); exit(-1); break;
      /*##switch-verify-beg##*/

      // code will be automatically inserted

      /*##switch-verify-end##*/
    }
  }
}


#ifdef USE_GPU
static void d_preprocess_encode(int& dpreencsize, byte*& dpreencdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = 0; i < prepros.size(); i++) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); exit(-1); break;
      /*##switch-device-preprocess-encode-beg##*/

      // code will be automatically inserted

      /*##switch-device-preprocess-encode-end##*/
    }
  }
}


static void d_preprocess_decode(int& dpredecsize, byte*& dpredecdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = prepros.size() - 1; i >= 0; i--) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); exit(-1); break;
      /*##switch-device-preprocess-decode-beg##*/

      // code will be automatically inserted

      /*##switch-device-preprocess-decode-end##*/
    }
  }
}
#endif


#ifdef USE_CPU
static void h_preprocess_encode(int& hpreencsize, byte*& hpreencdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = 0; i < prepros.size(); i++) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); exit(-1); break;
      /*##switch-host-preprocess-encode-beg##*/

      // code will be automatically inserted

      /*##switch-host-preprocess-encode-end##*/
    }
  }
}


static void h_preprocess_decode(int& hpredecsize, byte*& hpredecdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = prepros.size() - 1; i >= 0; i--) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); exit(-1); break;
      /*##switch-host-preprocess-decode-beg##*/

      // code will be automatically inserted

      /*##switch-host-preprocess-decode-end##*/
    }
  }
}
#endif


#ifdef USE_GPU
static void __global__ initBestSize(unsigned short* const bestSize, const int chunks)
{
  if ((threadIdx.x == 0) && (WS != warpSize)) {printf("ERROR: WS must be %d\n\n", warpSize); __trap();}  // debugging only
  for (int i = threadIdx.x; i < chunks; i += TPB) {
    bestSize[i] = CS;
  }
}


static void __global__ dbestChunkSize(const byte* const __restrict__ input, unsigned short* const __restrict__ bestSize)
{
  int* const head_in = (int*)input;
  const int outsize = head_in[0];
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  for (int chunkID = threadIdx.x; chunkID < chunks; chunkID += TPB) {
    bestSize[chunkID] = min(bestSize[chunkID], size_in[chunkID]);
  }
}


static void __global__ dcompareData(const int size, const byte* const __restrict__ data1, const byte* const __restrict__ data2, unsigned int* const __restrict__ min_loc)
{
  const int i = threadIdx.x + blockIdx.x * TPB;
  if (i < size) {
    if (data1[i] != data2[i]) atomicMin(min_loc, i);
  }
}


static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
}

static inline __device__ void propagate_carry(const int value, const int chunkID, volatile int* const __restrict__ fullcarry, int* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? value : -value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const int cidm1ml = chunkID - 1 - lane;
      int val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((__any_sync(~0, val == 0)) || (__all_sync(~0, val <= 0)));
#if defined(WS) && (WS == 64)
      const long long mask = __ballot_sync(~0, val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = __ballot_sync(~0, val > 0);
      const int pos = __ffs(mask) - 1;
#endif
      int partc = (lane < pos) ? -val : 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      partc = __reduce_add_sync(~0, partc);
#else
      partc += __shfl_xor_sync(~0, partc, 1);
      partc += __shfl_xor_sync(~0, partc, 2);
      partc += __shfl_xor_sync(~0, partc, 4);
      partc += __shfl_xor_sync(~0, partc, 8);
      partc += __shfl_xor_sync(~0, partc, 16);
#endif
      if (lane == pos) {
        const int fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}


// copy (len) bytes from global memory (source) to shared memory (destination) using separate shared memory buffer (temp)
// destination and temp must we word aligned, accesses up to CS + 3 bytes in temp
static inline __device__ void g2s(void* const __restrict__ destination, const void* const __restrict__ source, const int len, void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  if (len < 128) {
    byte* const __restrict__ output = (byte*)destination;
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      byte* const __restrict__ out = (byte*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    } else {
      const int offs = 4 - bcnt;  //(4 - bcnt) & 3;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      byte* const __restrict__ buffer = (byte*)temp;
      byte* const __restrict__ buf = (byte*)&buffer[offs];
      int* __restrict__ buf_w = (int*)&buffer[4];  //(int*)&buffer[(bcnt + 3) & 4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += TPB) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode(const unsigned long long chain, const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int* const __restrict__ outsize, int* const __restrict__ fullcarry)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;
  const int chunks = (insize + CS - 1) / CS;  // round up
  int* const head_out = (int*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = min(CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];

    // encode chunk
    int csize = osize;
    bool good = true;
    unsigned long long pipeline = chain;
    while ((pipeline != 0) && good) {
      __syncthreads();  // "out" produced, chunk[last] consumed
      byte* tmp = in; in = out; out = tmp;
      switch (pipeline & 0xff) {
        default: {byte* tmp = in; in = out; out = tmp;} break;
        /*##switch-device-encode-beg##*/

        // code will be automatically inserted

        /*##switch-device-encode-end##*/
      }
      pipeline >>= 8;
    }
    __syncthreads();  // "temp" and "out" done

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (int*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (int i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const int offs = (chunkID == 0) ? 0 : *((int*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_decode(const unsigned long long chain, const byte* const __restrict__ input, byte* const __restrict__ output, int* const __restrict__ g_outsize)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

  // input header
  int* const head_in = (int*)input;
  const int outsize = head_in[0];

  // initialize
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  int prevChunkID = 0;
  int prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    int sum = 0;
    for (int i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (int)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const int offs = prevOffset + block_sum_reduction(sum, (int*)&chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    byte* tmp = in; in = out; out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = min(CS, outsize - base);
    if (csize < osize) {
      unsigned long long pipeline = chain;
      while (pipeline != 0) {
        byte* tmp = in; in = out; out = tmp;
        switch (pipeline >> 56) {
          default: {byte* tmp = in; in = out; out = tmp;} break;
          /*##switch-device-decode-beg##*/

          // code will be automatically inserted

          /*##switch-device-decode-end##*/
        }
        __syncthreads();  // chunk transformed
        pipeline <<= 8;
      }
    }

    if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %d\n\n", csize, osize, chunkID); __trap();}
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}

#endif


#ifdef USE_CPU
static void h_encode(const unsigned long long chain, const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int& outsize)
{
  // initialize
  const int chunks = (insize + CS - 1) / CS;  // round up
  int* const head_out = (int*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  int* const carry = new int [chunks];
  memset(carry, 0, chunks * sizeof(int));

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const int base = chunkID * CS;
    const int osize = std::min(CS, insize - base);
    memcpy(out, &input[base], osize);

    // encode chunk
    int csize = osize;
    bool good = true;
    unsigned long long pipeline = chain;
    while ((pipeline != 0) && good) {
      std::swap(in, out);
      switch (pipeline & 0xff) {
        default: std::swap(in, out); break;
        /*##switch-host-encode-beg##*/

        // code will be automatically inserted

        /*##switch-host-encode-end##*/
      }
      pipeline >>= 8;
    }

    // handle carry and store chunk
    int offs = 0;
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
      carry[chunkID] = offs + csize;
      size_out[chunkID] = csize;
      memcpy(&data_out[offs], out, csize);
    } else {
      // store original data
      #pragma omp atomic write
      carry[chunkID] = offs + osize;
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


static void hbestChunkSize(const byte* const __restrict__ input, unsigned short* const __restrict__ bestSize)
{
  int* const head_in = (int*)input;
  const int outsize = head_in[0];
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    bestSize[chunkID] = std::min(bestSize[chunkID], size_in[chunkID]);
  }
}


static void h_decode(const unsigned long long chain, const byte* const __restrict__ input, byte* const __restrict__ output, int& outsize)
{
  // input header
  int* const head_in = (int*)input;
  outsize = head_in[0];

  // initialize
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];
  int* const start = new int [chunks];

  // convert chunk sizes into starting positions
  int pfs = 0;
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (int)size_in[chunkID];
  }

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const int base = chunkID * CS;
    const int osize = std::min(CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      // decompress
      memcpy(out, &data_in[start[chunkID]], csize);

      // decode
      unsigned long long pipeline = chain;
      while (pipeline != 0) {
        std::swap(in, out);
        switch (pipeline >> 56) {
          default: std::swap(in, out); break;
          /*##switch-host-decode-beg##*/

          // code will be automatically inserted

          /*##switch-host-decode-end##*/
        }
        pipeline <<= 8;
      }

      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %d\n\n", csize, osize, chunkID); exit(-1);}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}
#endif


#ifdef USE_GPU
struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}
#endif


#ifdef USE_CPU
struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};
#endif


static std::string getPipeline(unsigned long long pipeline, const int stages)
{
  std::string s;
  for (int i = 0; i < stages; i++) {
    switch (pipeline & 0xff) {
      default: s += " NUL"; break;
      /*##switch-pipeline-beg##*/

      // code will be automatically inserted

      /*##switch-pipeline-end##*/
    }
    pipeline >>= 8;
  }
  s.erase(0, 1);
  return s;
}


static std::map<std::string, byte> getPreproMap()
{
  std::map<std::string, byte> preprocessors;
  preprocessors["NUL"] = 0;
  /*##preprocessor-map-beg##*/

  // code will be automatically inserted

  /*##preprocessor-map-end##*/
  return preprocessors;
}


static std::string getPreprocessors(std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  std::string s;

  if (prepros.size() > 0) {
    const std::map<std::string, byte> prepro_name2num = getPreproMap();
    std::string prepro_num2name [256];
    for (auto pair: prepro_name2num) {
      prepro_num2name[pair.second] = pair.first;
    }

    for (int i = 0; i < prepros.size(); i++) {
      s += ' ';
      s += prepro_num2name[prepros[i].first];
      s += '(';
      bool first = true;
      for (double d: prepros[i].second) {
        if (first) {
          first = false;
        } else {
          s += ", ";
        }
        long long val = d;
        if (d == val) {
          s += std::to_string(val);
        } else {
          s += std::to_string(d);
        }
      }
      s += ')';
    }

    s.erase(0, 1);
  }

  return s;
}


static void printPreprocessors(FILE* f = stdout)
{
  const std::map<std::string, byte> prepro_name2num = getPreproMap();
  if (f == stdout) {
    fprintf(f, "%ld available preprocessors:\n", prepro_name2num.size());
    for (auto pair: prepro_name2num) {
      fprintf(f, "%s ", pair.first.c_str());
    }
    fprintf(f, "\n");
  } else {
    fprintf(f, "available preprocessors, %ld\n", prepro_name2num.size());
    for (auto pair: prepro_name2num) {
      fprintf(f, "%s\n", pair.first.c_str());
    }
  }
}


static std::map<std::string, byte> getCompMap()
{
  std::map<std::string, byte> components;
  components["NUL"] = 0;
  /*##component-map-beg##*/

  // code will be automatically inserted

  /*##component-map-end##*/
  return components;
}


static void printComponents(FILE* f = stdout)
{
  const std::map<std::string, byte> comp_name2num = getCompMap();
  if (f == stdout) {
    fprintf(f, "%ld available components:\n", comp_name2num.size());
    for (auto pair: comp_name2num) {
      fprintf(f, "%s ", pair.first.c_str());
    }
    fprintf(f, "\n");
  } else {
    fprintf(f, "available components, %ld\n", comp_name2num.size());
    for (auto pair: comp_name2num) {
      fprintf(f, "%s\n", pair.first.c_str());
    }
  }
}


static std::map<std::string, byte> getVerifMap()
{
  std::map<std::string, byte> verifs;
  /*##verifier-map-beg##*/

  // code will be automatically inserted

  /*##verifier-map-end##*/
  return verifs;
}


template <typename T>
static double Entropy(T *const data, const long long len)
{
  assert(sizeof(T) <= 2);
  const int size = 1 << (sizeof(T) * 8);
  long long hist [size];
  memset(hist, 0, size * sizeof(long long));
  for (long long i = 0; i < len; i++) {
    hist[data[i]]++;
  }
  double invtot = 1.0 / len;
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    if (hist[i] != 0) {
      double ent = hist[i] * invtot;
      sum += ent * log2(ent);
    }
  }
  return -sum;
}


template <typename T>
static double entropy(const T* const data, const long long len)
{
  double sum = 0.0;
  if (len > 0) {
    T* const copy = new T [len];
    for (long long i = 0; i < len; i++) copy[i] = data[i];
    std::sort(&copy[0], &copy[len]);

    const double invlen = 1.0 / len;
    long long cnt = 1;
    T prev = copy[0];
    for (long long i = 1; i < len; i++) {
      if (copy[i] == prev) {
        cnt++;
      } else {
        const double ent = cnt * invlen;
        sum += ent * log2(ent);
        cnt = 1;
        prev = copy[i];
      }
    }
    const double ent = cnt * invlen;
    sum += ent * log2(ent);
    sum = -sum;
    delete [] copy;
  }
  return sum;
}


template <typename T>
static void Frequency(const T* const data, const int len)
{
  assert(sizeof(T) <= 2);
  const int size = 1 << (sizeof(T) * 8);
  long long hist [size];
  memset(hist, 0, size * sizeof(long long));
  for (long long i = 0; i < len; i++) {
    hist[data[i]]++;
  }
  std::vector<std::pair<long long, T>> vec;
  for (int i = 0; i < size; i++) {
    if (hist[i] != 0) {
      vec.push_back(std::make_pair(-hist[i], (T)i));
    }
  }

  printf(" unique values: %ld\n", vec.size());
  printf(" occurrences\n");
  std::sort(vec.begin(), vec.end());
  for (int i = 0; i < std::min(8, (int)vec.size()); i++) {
    printf(" %14lld: %14lld  (%6.3f%%)\n", (long long)vec[i].second, -vec[i].first, -100.0 * vec[i].first / len);
  }
}


template <typename T>
static void frequency(const T* const data, const int len)
{
  std::vector<std::pair<int, T>> vec;
  if (len > 0) {
    T* const copy = new T [len];
    for (int i = 0; i < len; i++) copy[i] = data[i];
    std::sort(&copy[0], &copy[len]);

    int cnt = 1;
    T prev = copy[0];
    for (int i = 1; i < len; i++) {
      if (copy[i] == prev) {
        cnt++;
      } else {
        vec.push_back(std::make_pair(-cnt, prev));
        cnt = 1;
        prev = copy[i];
      }
    }
    vec.push_back(std::make_pair(-cnt, prev));
    delete [] copy;
  }

  printf(" unique values: %ld\n", vec.size());
  printf(" occurrences\n");
  std::sort(vec.begin(), vec.end());
  for (int i = 0; i < std::min(8, (int)vec.size()); i++) {
    printf(" %20lld: %20d  (%6.3f%%)\n", (long long)vec[i].second, -vec[i].first, -100.0 * vec[i].first / len);
  }
}


struct Elem {
  unsigned long long pipe;
  float CR;
  float HencThru;
  float HdecThru;
  float DencThru;
  float DdecThru;
};


#ifdef USE_GPU
static bool compareElemDencThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.DencThru < e2.DencThru));
}


static bool compareElemDdecThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.DdecThru < e2.DdecThru));
}
#endif


#ifdef USE_CPU
static bool compareElemHencThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.HencThru < e2.HencThru));
}


static bool compareElemHdecThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.HdecThru < e2.HdecThru));
}
#endif


static std::vector<std::pair<byte, std::vector<double>>> getItems(std::map<std::string, byte> item_name2num, char* const s)
{
  std::vector<std::pair<byte, std::vector<double>>> items;

  char* p = s;
  while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
  while (*p != 0) {
    // get name
    char* beg = p;
    while ((*p != 0) && (*p != ' ') && (*p != '\t') && (*p != '(')) p++;  // find end of name
    char* end = p;
    if (end <= beg) {fprintf(stderr, "ERROR: expected an item name in specification\n\n"); exit(-1);}
    char old = *end;
    *end = 0;  // string terminator
    int num = -1;
    for (auto pair: item_name2num) {
      const std::string itemname = pair.first;
      const byte itemnum = pair.second;
      if (itemname.compare(beg) == 0) {
        num = itemnum;
        break;
      }
    }
    if (num < 0) {fprintf(stderr, "ERROR: unknown item name\n\n"); exit(-1);}
    *end = old;

    // read in parameters
    std::vector<double> params;
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    if (*p != '(') {fprintf(stderr, "ERROR: expected '(' in specification\n\n"); exit(-1);}
    p++;
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    while ((*p != 0) && (*p != ')')) {
      // get double
      char* pos;
      const double d = std::strtod(p, &pos);
      if (pos == p) {fprintf(stderr, "ERROR: expected a value in specification\n\n"); exit(-1);}
      p = pos;
      params.push_back(d);
      while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
      if (*p == ')') break;

      // consume comma
      if (*p != ',') {fprintf(stderr, "ERROR: expected ',' in specification\n\n"); exit(-1);}
      p++;
      while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    }
    if (*p != ')') {fprintf(stderr, "ERROR: expected ')' in specification\n\n"); exit(-1);}
    p++;
    items.push_back(std::make_pair((byte)num, params));
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
  }

  return items;
}


static std::vector<std::vector<byte>> getStages(std::map<std::string, byte> comp_name2num, char* const regex, int& stages, unsigned long long& algorithms)
{
  std::vector<std::vector<byte>> comp_list;

  int s = 0;
  char* ptr = strtok(regex, " \t");
  while (ptr != NULL) {
    if (s >= max_stages) {fprintf(stderr, "ERROR: number of stages must be between 1 and %d\n\n", max_stages); exit(-1);}

    std::vector<byte> list;
    std::string in = ptr;
    const bool inv = (in[0] == '~');
    if (inv) in = in.substr(1);
    const std::regex re(in);
    for (auto pair: comp_name2num) {
      const std::string compname = pair.first;
      const byte compnum = pair.second;
      if (std::regex_match(compname, re)) {
        if (!inv) list.push_back(compnum);
      } else {
        if (inv) list.push_back(compnum);
      }
    }
    comp_list.push_back(list);
    s++;
    ptr = strtok(NULL, " \t");
  }

  stages = s;
  if (stages < 1) {fprintf(stderr, "ERROR: stages must be between 1 and %d\n\n", max_stages); exit(-1);}

  algorithms = 1;
  for (int s = 0; s < stages; s++) {
    algorithms *= comp_list[s].size();
  }

  return comp_list;
}


static void printStages(std::vector<std::pair<byte, std::vector<double>>> prepros, std::map<std::string, byte> prepro_name2num, std::vector<std::vector<byte>> comp_list, std::map<std::string, byte> comp_name2num, const int stages, const unsigned long long algorithms, FILE* f = stdout)
{
  std::string prepro_num2name [256];
  for (auto pair: prepro_name2num) {
    prepro_num2name[pair.second] = pair.first;
  }

  std::string comp_num2name [256];
  for (auto pair: comp_name2num) {
    comp_num2name[pair.second] = pair.first;
  }

  int max = 0;
  for (int s = 0; s < stages; s++) {
    max = std::max(max, (int)comp_list[s].size());
  }

  if (f == stdout) {
    printf("algorithms: %lld\n\n", algorithms);
    if (prepros.size() > 0) {
      printf("  preprocessors\n  -------------\n");
      for (int i = 0; i < prepros.size(); i++) {
        printf("  %s(", prepro_num2name[prepros[i].first].c_str());
        bool first = true;
        for (double d: prepros[i].second) {
          if (first) {
            first = false;
          } else {
            printf(", ");
          }
          long long val = d;
          if (d == val) {
            printf("%lld", val);
          } else {
            printf("%e", d);
          }
        }
        printf(")");
      }
      printf("\n\n");
    }

    for (int s = 0; s < stages; s++) printf("  stage %d", s + 1);
    printf("\n");
    for (int s = 0; s < stages; s++) printf("  -------");
    printf("\n");
    for (int e = 0; e < max; e++) {
      for (int s = 0; s < stages; s++) {
        if (e < comp_list[s].size()) {
          printf("%9s", comp_num2name[comp_list[s][e]].c_str());
        } else {
          printf("%9s", "");
        }
      }
      printf("\n");
    }
    printf("\n");
  } else {
    fprintf(f, "algorithms, %lld\n\n", algorithms);
    if (prepros.size() > 0) {
      fprintf(f, "preprocessors\n");
      for (int i = 0; i < prepros.size(); i++) {
        fprintf(f, "%s", prepro_num2name[prepros[i].first].c_str());
        for (double d: prepros[i].second) {
          long long val = d;
          if (d == val) {
            fprintf(f, ", %lld", val);
          } else {
            fprintf(f, ", %e", d);
          }
        }
        fprintf(f, "\n");
      }
      fprintf(f, "\n");
    }

    for (int s = 0; s < stages; s++) fprintf(f, "stage %d, ", s + 1);
    fprintf(f, "\n");
    for (int e = 0; e < max; e++) {
      for (int s = 0; s < stages; s++) {
        if (e < comp_list[s].size()) {
          fprintf(f, "%s, ", comp_num2name[comp_list[s][e]].c_str());
        } else {
          fprintf(f, ", ");
        }
      }
      fprintf(f, "\n");
    }
    fprintf(f, "\n");
  }
}


static void printUsage(char* argv [])
{
  printf("USAGE: %s input_file_name AL \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\" [\"verifier\"]\n", argv[0]);
  printf("USAGE: %s input_file_name PR \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\" [\"verifier\"]\n", argv[0]);
  printf("USAGE: %s input_file_name CR \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\"\n", argv[0]);
  printf("USAGE: %s input_file_name EX \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\" [\"verifier\"]\n", argv[0]);
  printf("USAGE: %s input_file_name TS\n", argv[0]);
  printf("\n");
  printPreprocessors();
  printf("\n");
  printComponents();
  printf("\nFor usage examples, please see the quick-start guide and tutorial at https://github.com/burtscher/LC-framework/.\n");
/*
  printf("\nExamples:\n");
  printf("1. Lossless 2-stage pipeline with CLOG in 2nd stage using 4-byte granularity, showing only compression ratio (CR):\n\n   ./lc input_file_name CR \"\" \".+ CLOG_4\"\n\n");
  printf("2. Lossless 3-stage pipeline with 3D Lorenzo preprocessor and a DIFF, open, and R2E stage using 8-byte granularity, showing compression ratio, compression and decompression throughput, and Pareto frontier (EX):\n\n   ./lc input_file_name EX \"LOR3D_i32(dim1, dim2, dim3)\" \"DIFF_8 .+ R2E_8\"\n\n");
  printf("3. Lossy 2-stage pipeline with quantization with a 0.001 error bound and a limit value of 1000 and a CLOG component using 4-byte granularity, showing only compression ratio:\n\n  ./lc input_file_name CR \"QUANT_ABS_R_f32(0.001, 1000)\" \"CLOG_4\"\n\n");
  printf("Notes:\n1. The double quotations are always needed, even if there is nothing between them.\n2. The Lorenzo preprocessors only work when the passed dimensions match the input file size.\n");
  printf("3. The quantization preprocessors always need an error bound parameter specified in parentheses.\n");
  printf("4. The quantization preprocessors optionally take a second value, which indicates the absolute value beyond which the values are compressed losslessly.\n");
  printf("5. See the ./verifiers directory for a list of available verifiers. Verifiers take an error bound as parameter.\n\n");
*/
}


struct Config {
  bool speed;  // print speed info
  bool size;  // print size info
  bool warmup;  // perform warmup run
  bool memcopy;  // measure memcopy speed
  bool decom;  // perform decompression
  bool verify;  // verify results
  bool csv;  // output CSV file
};


int main(int argc, char* argv [])
{
  printf("LC Compression Framework v1.2 (%s)\n", __FILE__);

#ifndef USE_GPU
  #ifndef USE_CPU
  fprintf(stderr, "ERROR: must define 'USE_CPU', 'USE_GPU', or both when compiling code\n\n");
  exit(-1);
  #else
  printf("CPU version\n");
  #endif
#else
  #ifndef USE_CPU
  printf("GPU version\n");
  #else
  printf("Combined CPU + GPU version\n");
  #endif
#endif

  printf("Copyright 2024 Texas State University\n\n");

  // perform system checks
  if (CS % 8 != 0) {fprintf(stderr, "ERROR: CS must be a multiple of 8\n\n"); exit(-1);}
  const int endian = 1;
  if (*((char*)(&endian)) != 1) {fprintf(stderr, "ERROR: framework only supports little-endian systems\n\n"); exit(-1);}
  if (sizeof(long long) != 8) {fprintf(stderr, "ERROR: long long must be 8 bytes\n\n"); exit(-1);}
  if (sizeof(int) != 4) {fprintf(stderr, "ERROR: int must be 4 bytes\n\n"); exit(-1);}
  if (sizeof(short) != 2) {fprintf(stderr, "ERROR: short must be 2 bytes\n\n"); exit(-1);}
  if (sizeof(char) != 1) {fprintf(stderr, "ERROR: char must be 1 byte\n\n"); exit(-1);}

  // print usage message if needed
  if ((argc < 3) || (argc > 6) || (argc == 4)) {
    printUsage(argv);
    exit(-1);
  }

  // generate preprocessor maps
  std::map<std::string, byte> prepro_name2num = getPreproMap();
  std::string prepro_num2name [256];
  for (auto pair: prepro_name2num) {
    prepro_num2name[pair.second] = pair.first;
  }

  // generate component maps
  std::map<std::string, byte> comp_name2num = getCompMap();
  std::string comp_num2name [256];
  for (auto pair: comp_name2num) {
    comp_num2name[pair.second] = pair.first;
  }

  // generate verifier maps
  std::map<std::string, byte> verif_name2num = getVerifMap();
  std::string verif_num2name [256];
  for (auto pair: verif_name2num) {
    verif_num2name[pair.second] = pair.first;
  }

  // read command line
  Config conf;  // speed, size, warmup, memcopy, decom, verify, csv
  int stages;
  unsigned long long algorithms;
  std::vector<std::pair<byte, std::vector<double>>> prepros;
  std::vector<std::vector<byte>> comp_list;
  std::vector<std::pair<byte, std::vector<double>>> verifs;
  std::vector<double> dummy;
  verifs.push_back(std::make_pair((byte)v_LOSSLESS, dummy));
  std::string ext = ".null";
  if (strcmp(argv[2], "TS") == 0) {  // verify all pairs, no metrics
    if (argc != 3) {printUsage(argv); exit(-1);}
    char regex [] = {'.', '+', ' ', '.', '+', 0};
    comp_list = getStages(comp_name2num, regex, stages, algorithms);
    conf = {false, false, false, false, true, true, false};  // -speed, -size, -warmup, -memcopy, +decom, +verify, -csv
  } else if (strcmp(argv[2], "CR") == 0) {  // exhaustive with only CR, no speed
    if (argc != 5) {printUsage(argv); exit(-1);}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (algorithms < 1) {fprintf(stderr, "ERROR: need at least one algorithm\n\n"); exit(-1);}
    conf = {false, false, false, false, false, false, true};  // -speed, -size, -warmup, -memcopy, -decom, -verify, +csv
    ext = ".CR" + std::to_string(stages);
  } else if (strcmp(argv[2], "EX") == 0) {  // exhaustive with all metrics
    if ((argc != 5) && (argc != 6)) {printUsage(argv); exit(-1);}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (argc == 6) verifs = getItems(verif_name2num, argv[5]);
    if (algorithms < 1) {fprintf(stderr, "ERROR: need at least one algorithm\n\n"); exit(-1);}
    conf = {false, false, true, false, true, true, true};  // -speed, -size, +warmup, -memcopy, +decom, +verify, +csv
    ext = ".EX" + std::to_string(stages);
  } else if (strcmp(argv[2], "PR") == 0) {  // just print stages and exit
    if ((argc != 5) && (argc != 6)) {printUsage(argv); exit(-1);}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (argc == 6) verifs = getItems(verif_name2num, argv[5]);
    printStages(prepros, prepro_name2num, comp_list, comp_name2num, stages, algorithms);
    return 0;
  } else if (strcmp(argv[2], "AL") == 0) {  // single algorithm with all metrics
    if ((argc != 5) && (argc != 6)) {printUsage(argv); exit(-1);}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (argc == 6) verifs = getItems(verif_name2num, argv[5]);
    if (algorithms != 1) {fprintf(stderr, "ERROR: pipeline must describe one algorithm\n\n"); exit(-1);}
    conf = {true, true, true, true, true, true, false};  // +speed, +size, +warmup, +memcopy, +decom, +verify, -csv
  } else {  // unknown mode
    printUsage(argv);
    exit(-1);
  }
  printStages(prepros, prepro_name2num, comp_list, comp_name2num, stages, algorithms);

  // read input file
  printf("input: %s\n", argv[1]);
  FILE* const fin = fopen(argv[1], "rb"); assert(fin != NULL);
  fseek(fin, 0, SEEK_END);
  const int fsize = ftell(fin); assert(fsize > 0);
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(input, 1, fsize, fin); assert(insize == fsize);
  fclose(fin);
  printf("input size: %d bytes\n\n", insize);

#ifdef USE_GPU
  // copy input to GPU
  byte* d_input;
  cudaMalloc((void **)&d_input, insize);
  cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);

  // get and print GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  const float bw = 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) * 0.000001;
  printf("     %.1f GB/s (%.1f+%.1f) peak bandwidth (%d-bit bus)\n\n", bw, bw / 2, bw / 2, deviceProp.memoryBusWidth);
  const int blocks = SMs * (mTpSM / TPB);
  CheckCuda(__LINE__);

  // time GPU memcpy (for reference only)
  double dmcthroughput = std::numeric_limits<double>::infinity();
  if (conf.memcopy) {
    byte* d_data;
    cudaMalloc((void **)&d_data, insize);
    if (conf.warmup) {
      cudaMemcpy(d_data, d_input, insize, cudaMemcpyDeviceToDevice);
    }
    GPUTimer dtimer;
    dtimer.start();
    cudaMemcpy(d_data, d_input, insize, cudaMemcpyDeviceToDevice);
    const double druntime = dtimer.stop();
    dmcthroughput = insize * 0.000000001 / druntime;
    if (conf.speed) {
      printf("GPU memcpy runtime: %.6f s\n", druntime);
      printf("GPU memory rd throughput:          %8.3f Gbytes/s %8.1f%%\n", dmcthroughput, 100.0);
      printf("GPU memory wr throughput:          %8.3f Gbytes/s %8.1f%%\n\n", dmcthroughput, 100.0);
    }
    cudaFree(d_data);
    CheckCuda(__LINE__);
  }
#endif

#ifdef USE_CPU
  // time CPU memcpy (for reference only)
  double hmcthroughput = std::numeric_limits<double>::infinity();
  if (conf.memcopy) {
    byte* data = new byte [insize];
    if (conf.warmup) {
      memcpy(data, input, insize);
    }
    CPUTimer htimer;
    htimer.start();
    memcpy(data, input, insize);
    const double hruntime = htimer.stop();
    hmcthroughput = insize * 0.000000001 / hruntime;
    if (conf.speed) {
      printf("CPU memcpy runtime: %.6f s\n", hruntime);
      printf("CPU memory rd throughput:          %8.3f Gbytes/s %8.1f%%\n", hmcthroughput, 100.0);
      printf("CPU memory wr throughput:          %8.3f Gbytes/s %8.1f%%\n\n", hmcthroughput, 100.0);
    }
    delete [] data;
  }
#endif

#ifdef USE_GPU
  // time GPU preprocessor encoding
  byte* d_preencdata;
  if (conf.warmup) {
    cudaMalloc((void **)&d_preencdata, insize);
    cudaMemcpy(d_preencdata, d_input, insize, cudaMemcpyDeviceToDevice);
    int dpreencsize = insize;
    d_preprocess_encode(dpreencsize, d_preencdata, prepros);
    cudaFree(d_preencdata);
  }
  cudaMalloc((void **)&d_preencdata, insize);
  cudaMemcpy(d_preencdata, d_input, insize, cudaMemcpyDeviceToDevice);
  int dpreencsize = insize;
  GPUTimer dtimer;
  dtimer.start();
  d_preprocess_encode(dpreencsize, d_preencdata, prepros);
  const double dpreenctime = (prepros.size() == 0) ? 0 : dtimer.stop();
  if (conf.speed) printf("GPU preprocessor encoding time: %.6f s\n", dpreenctime);
  if (conf.size) printf("GPU preprocessor encoded size: %d bytes\n\n", dpreencsize);

  // time GPU preprocessor decoding
  int dpredecsize = 0;
  double dpredectime = 0;
  byte* d_predecdata = NULL;
  if (conf.decom) {
    if (conf.warmup) {
      cudaMalloc((void **)&d_predecdata, dpreencsize);
      cudaMemcpy(d_predecdata, d_preencdata, dpreencsize, cudaMemcpyDeviceToDevice);
      dpredecsize = dpreencsize;
      d_preprocess_decode(dpredecsize, d_predecdata, prepros);
      cudaFree(d_predecdata);
    }
    cudaMalloc((void **)&d_predecdata, dpreencsize);
    cudaMemcpy(d_predecdata, d_preencdata, dpreencsize, cudaMemcpyDeviceToDevice);
    dpredecsize = dpreencsize;
    dtimer.start();
    d_preprocess_decode(dpredecsize, d_predecdata, prepros);
    dpredectime = (prepros.size() == 0) ? 0 : dtimer.stop();
    CheckCuda(__LINE__);
    if (conf.speed) printf("GPU preprocessor decoding time: %.6f s\n", dpredectime);
    if (conf.size) printf("GPU preprocessor decoded size: %d bytes\n\n", dpredecsize);
  }
#endif

#ifdef USE_CPU
  // time CPU preprocessor encoding
  if (conf.warmup) {
    byte* hpreencdata = new byte [insize];
    std::copy(input, input + insize, hpreencdata);
    int hpreencsize = insize;
    h_preprocess_encode(hpreencsize, hpreencdata, prepros);
    delete [] hpreencdata;
  }
  byte* hpreencdata = new byte [insize];
  std::copy(input, input + insize, hpreencdata);
  int hpreencsize = insize;
  CPUTimer htimer;
  htimer.start();
  h_preprocess_encode(hpreencsize, hpreencdata, prepros);
  const double hpreenctime = (prepros.size() == 0) ? 0 : htimer.stop();
  if (conf.speed) printf("CPU preprocessor encoding time: %.6f s\n", hpreenctime);
  if (conf.size) printf("CPU preprocessor encoded size: %d bytes\n\n", hpreencsize);

  // time CPU preprocessor decoding
  int hpredecsize = 0;
  double hpredectime = 0;
  byte* hpredecdata = NULL;
  if (conf.decom) {
    if (conf.warmup) {
      byte* hpredecdata = new byte [hpreencsize];
      std::copy(hpreencdata, hpreencdata + hpreencsize, hpredecdata);
      hpredecsize = hpreencsize;
      h_preprocess_decode(hpredecsize, hpredecdata, prepros);
      delete [] hpredecdata;
    }
    hpredecdata = new byte [hpreencsize];
    std::copy(hpreencdata, hpreencdata + hpreencsize, hpredecdata);
    hpredecsize = hpreencsize;
    htimer.start();
    h_preprocess_decode(hpredecsize, hpredecdata, prepros);
    hpredectime = (prepros.size() == 0) ? 0 : htimer.stop();
    if (conf.speed) printf("CPU preprocessor decoding time: %.6f s\n", hpredectime);
    if (conf.size) printf("CPU preprocessor decoded size: %d bytes\n\n", hpredecsize);
  }
#endif

  // verify results
  if (conf.verify && conf.decom) {
    bool bug = false;

#ifdef USE_GPU
    if (dpredecsize != insize) {fprintf(stderr, "ERROR: dpredec size wrong: is %d instead of %d\n\n", dpredecsize, insize); bug = true;}
    byte* dpredecdata = new byte [dpredecsize];
    cudaMemcpy(dpredecdata, d_predecdata, dpredecsize, cudaMemcpyDeviceToHost);
    verify(std::min(insize, dpredecsize), dpredecdata, input, verifs);
    if (!bug && (algorithms == 1)) {
      //FILE* const f = fopen("LC.decoded", "wb");  assert(f != NULL);
      //const int num = fwrite(dpredecdata, sizeof(byte), dpredecsize, f);  assert(num == dpredecsize);
      //fclose(f);
    }
    delete [] dpredecdata;
#endif

#ifdef USE_CPU
    if (hpredecsize != insize) {fprintf(stderr, "ERROR: hpredec size wrong: is %d instead of %d\n\n", hpredecsize, insize); bug = true;}
    verify(std::min(insize, hpredecsize), hpredecdata, input, verifs);
#ifndef USE_GPU
    if (!bug && (algorithms == 1)) {
      //FILE* const f = fopen("LC.decoded", "wb");  assert(f != NULL);
      //const int num = fwrite(hpredecdata, sizeof(byte), hpredecsize, f);  assert(num == hpredecsize);
      //fclose(f);
    }
#endif
#endif

#ifdef USE_GPU
#ifdef USE_CPU
    byte* dpreencdata = new byte [dpreencsize];
    cudaMemcpy(dpreencdata, d_preencdata, dpreencsize, cudaMemcpyDeviceToHost);
    if (dpreencsize != hpreencsize) {fprintf(stderr, "ERROR: hpreencsize and dpreencsize differ: %d vs %d\n\n", hpreencsize, dpreencsize); bug = true;}
    for (int i = 0; i < std::min(hpreencsize, dpreencsize); i++) {
      if (dpreencdata[i] != hpreencdata[i]) {
        fprintf(stderr, "ERROR: CPU and GPU preprocessor encoded results differ at pos %d: %x vs %x\n\n", i, hpreencdata[i], dpreencdata[i]);
        bug = true;
        break;
      }
    }
    delete [] dpreencdata;
    printf("compared preenc CPU to GPU\n");  //MB: remove later

    dpredecdata = new byte [dpredecsize];
    cudaMemcpy(dpredecdata, d_predecdata, dpredecsize, cudaMemcpyDeviceToHost);
    if (dpredecsize != hpredecsize) {fprintf(stderr, "ERROR: hpredecsize and dpredecsize differ: %d vs %d\n\n", hpredecsize, dpredecsize); bug = true;}
    for (int i = 0; i < std::min(hpredecsize, dpredecsize); i++) {
      if (dpredecdata[i] != hpredecdata[i]) {
        fprintf(stderr, "ERROR: CPU and GPU preprocessor decoded results differ at pos %d: %x vs %x\n\n", i, hpredecdata[i], dpredecdata[i]);
        bug = true;
        break;
      }
    }
    delete [] dpredecdata;
    printf("compared predec CPU to GPU\n");  //MB: remove later
#endif
#endif

    if (bug) {
      exit(-1);
    } else {
      printf("preprocessor verification passed\n\n");
    }
  }

  // write header to CSV file (if enabled)
  FILE* fres = NULL;
  if (conf.csv) {
    std::string fname(argv[1]);
    size_t sep = fname.find_last_of("\\/");
    if (sep != std::string::npos) fname = fname.substr(sep + 1, fname.size() - sep - 1);
    fname += ext + ".csv";
    fres = fopen(fname.c_str(), "wt"); assert(fres != NULL);
    fprintf(fres, "LC Lossless Compression Framework v1.2\n");
    fprintf(fres, "input, %s\n", argv[1]);
    fprintf(fres, "size [bytes], %d\n", insize);
    time_t curtime;
    time(&curtime);
    fprintf(fres, "date/time, %s", ctime(&curtime));
    char name [256];
    gethostname(name, sizeof(name));
    fprintf(fres, "host name, %s\n", name);

#ifdef USE_GPU
    fprintf(fres, "GPU, %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
    fprintf(fres, ", %.1f (%.1f+%.1f) GB/s peak bandwidth (%d-bit bus)\n\n", bw, bw / 2, bw / 2, deviceProp.memoryBusWidth);
#endif

    printStages(prepros, prepro_name2num, comp_list, comp_name2num, stages, algorithms, fres);
    printComponents(fres);
    fprintf(fres, "\n");

    if (prepros.size() > 0) {
#ifdef USE_CPU
      fprintf(fres, "host preprocessor encoding time [s], host preprocessor decoding time [s], ");
#endif
#ifdef USE_GPU
      fprintf(fres, "device preprocessor encoding  time [s], device preprocessor decoding time [s]");
#endif
      fprintf(fres, "\n");
#ifdef USE_CPU
      fprintf(fres, "%f, %f, ", hpreenctime, hpredectime);
#endif
#ifdef USE_GPU
      fprintf(fres, "%f, %f, ", dpreenctime, dpredectime);
#endif
      fprintf(fres, "\n\n");
    }

    fprintf(fres, "algorithm, compressed size [bytes]");
    if (conf.decom) {
#ifdef USE_CPU
      fprintf(fres, ", host compression time [s], host decompression time [s]");
#endif
#ifdef USE_GPU
      fprintf(fres, ", device compression time [s], device decompression time [s]");
#endif
    }
    fprintf(fres, "\n");
  }

#ifdef USE_GPU
  // allocate GPU memory
  const int dchunks = (dpreencsize + CS - 1) / CS;  // round up
  const int dmaxsize = 3 * sizeof(int) + dchunks * sizeof(short) + dchunks * CS;  //MB: adjust later
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, dmaxsize);
  int* d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(int));
  byte* d_decoded;
  cudaMalloc((void **)&d_decoded, dpreencsize);
  int* d_decsize;
  cudaMalloc((void **)&d_decsize, sizeof(int));
  CheckCuda(__LINE__);
#endif

#ifdef USE_CPU
  // allocate CPU memory
  const int hchunks = (hpreencsize + CS - 1) / CS;  // round up
  const int hmaxsize = 3 * sizeof(int) + hchunks * sizeof(short) + hchunks * CS;  //MB: adjust later
  byte* const hencoded = new byte [hmaxsize];
  int hencsize = 0;
  byte* const hdecoded = new byte [hpreencsize];
  int hdecsize = 0;
#endif

#ifdef USE_GPU
  // run GPU experiments
  float dbestCR = 100.0;
  unsigned long long dbestPipe = 0;
  int dbestEncSize = insize;
  unsigned short* d_bestSize;
  cudaMalloc((void **)&d_bestSize, sizeof(unsigned short) * dchunks);
  initBestSize<<<1, TPB>>>(d_bestSize, dchunks);
  float denctime = -1;
  float ddectime = -1;
#endif

#ifdef USE_CPU
  // run CPU experiments
  float hbestCR = 100.0;
  unsigned long long hbestPipe = 0;
  int hbestEncSize = insize;
  unsigned short* const hbestSize = new unsigned short [hchunks];
  for (int i = 0; i < hchunks; i++) hbestSize[i] = CS;
  float henctime = -1;
  float hdectime = -1;
#endif

  // declare variables to hold stats
  std::vector<Elem> data;

  if (algorithms > 0) {
    unsigned long long combin = 0;
    int carrypos;
    do {
      // create chain for current combination and output
      unsigned long long chain = 0;
      for (int s = 0; s < stages; s++) {
        unsigned long long compnum = comp_list[s][(combin >> (s * 8)) & 0xff];
        chain |= compnum << (s * 8);
      }

      if (conf.csv) fprintf(fres, "%s, ", getPipeline(chain, stages).c_str());
      printf("pipeline: %s\n", getPipeline(chain, stages).c_str());

#ifdef USE_GPU
      if (conf.verify && conf.decom) {
        cudaMemset(d_encoded, -1, dmaxsize);  //MB: for testing only
        cudaMemset(d_decoded, -1, insize);  //MB: for testing only
      }
      CheckCuda(__LINE__);
#endif

#ifdef USE_CPU
      if (conf.verify && conf.decom) {
        memset(hencoded, -1, hmaxsize);  //MB: for testing only
        memset(hdecoded, -1, insize);  //MB: for testing only
      }
#endif

#ifdef USE_GPU
      // time GPU encoding
      if (conf.warmup) {
        int* d_fullcarry;
        cudaMalloc((void **)&d_fullcarry, dchunks * sizeof(int));
        d_reset<<<1, 1>>>();
        cudaMemset(d_fullcarry, 0, dchunks * sizeof(int));
        d_encode<<<blocks, TPB>>>(chain, d_preencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry);
        cudaFree(d_fullcarry);
        cudaDeviceSynchronize();
        CheckCuda(__LINE__);
      }
      GPUTimer dtimer;
      dtimer.start();
      int* d_fullcarry;
      cudaMalloc((void **)&d_fullcarry, dchunks * sizeof(int));
      d_reset<<<1, 1>>>();
      cudaMemset(d_fullcarry, 0, dchunks * sizeof(int));
      d_encode<<<blocks, TPB>>>(chain, d_preencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry);
      cudaFree(d_fullcarry);
      cudaDeviceSynchronize();
      denctime = dtimer.stop() + dpreenctime;
      if (conf.speed) printf("GPU encoding time: %.6f s\n", denctime);
      double dthroughput = insize * 0.000000001 / denctime;
      if (conf.speed) printf("GPU encoding memory rd throughput: %8.3f Gbytes/s %8.1f%%\n", dthroughput, 100.0 * dthroughput / dmcthroughput);
      CheckCuda(__LINE__);

      // get encoded GPU result
      int dencsize = 0;
      cudaMemcpy(&dencsize, d_encsize, sizeof(int), cudaMemcpyDeviceToHost);
      dthroughput = dencsize * 0.000000001 / denctime;
      if (conf.speed) printf("GPU encoding memory wr throughput: %8.3f Gbytes/s %8.1f%%\n", dthroughput, 100.0 * dthroughput / dmcthroughput);
      if (conf.size) printf("GPU encoded size: %d bytes\n", dencsize);
      CheckCuda(__LINE__);

      dbestChunkSize<<<1, TPB>>>(d_encoded, d_bestSize);

      // time GPU decoding
      int ddecsize = 0;
      if (conf.decom) {
        if (conf.warmup) {
          d_reset<<<1, 1>>>();
          d_decode<<<blocks, TPB>>>(chain, d_encoded, d_decoded, d_decsize);
          cudaDeviceSynchronize();
          CheckCuda(__LINE__);
        }
        dtimer.start();
        d_reset<<<1, 1>>>();
        unsigned long long schain = chain;
        if (chain != 0) {
          while ((schain >> 56) == 0) schain <<= 8;
        }
        d_decode<<<blocks, TPB>>>(schain, d_encoded, d_decoded, d_decsize);
        cudaDeviceSynchronize();
        ddectime = dtimer.stop() + dpredectime;
        if (conf.speed) printf("GPU decoding time: %.6f s\n", ddectime);
        dthroughput = dencsize * 0.000000001 / ddectime;
        if (conf.speed) printf("GPU decoding memory rd throughput: %8.3f Gbytes/s %8.1f%%\n", dthroughput, 100.0 * dthroughput / dmcthroughput);
        dthroughput = insize * 0.000000001 / ddectime;
        if (conf.speed) printf("GPU decoding memory wr throughput: %8.3f Gbytes/s %8.1f%%\n", dthroughput, 100.0 * dthroughput / dmcthroughput);
        CheckCuda(__LINE__);

        // get decoded GPU result
        cudaMemcpy(&ddecsize, d_decsize, sizeof(int), cudaMemcpyDeviceToHost);
        if (conf.size) printf("GPU decoded size: %d bytes\n", ddecsize);
        CheckCuda(__LINE__);
      }

      // print experiment result and record if best
      const float dCR = (100.0 * dencsize) / insize;
      printf("compression: %6.2f%% %7.3fx  (%d bytes)\n", dCR, 100.0 / dCR, dencsize);
      if (dbestCR > dCR) {
        dbestCR = dCR;
        dbestPipe = chain;
        dbestEncSize = dencsize;
      }
#endif

#ifdef USE_CPU
      // time CPU encoding
      if (conf.warmup) {
        h_encode(chain, hpreencdata, hpreencsize, hencoded, hencsize);
      }
      CPUTimer htimer;
      htimer.start();
      h_encode(chain, hpreencdata, hpreencsize, hencoded, hencsize);
      henctime = htimer.stop() + hpreenctime;
      if (conf.speed) printf("CPU encoding time: %.6f s\n", henctime);
      double hthroughput = insize * 0.000000001 / henctime;
      if (conf.speed) printf("CPU encoding memory rd throughput: %8.3f Gbytes/s %8.1f%%\n", hthroughput, 100.0 * hthroughput / hmcthroughput);
      hthroughput = hencsize * 0.000000001 / henctime;
      if (conf.speed) printf("CPU encoding memory wr throughput: %8.3f Gbytes/s %8.1f%%\n", hthroughput, 100.0 * hthroughput / hmcthroughput);
      if (conf.size) printf("CPU encoded size: %d bytes\n", hencsize);

      hbestChunkSize(hencoded, hbestSize);

      // time CPU decoding
      if (conf.decom) {
        if (conf.warmup) {
          h_decode(chain, hencoded, hdecoded, hdecsize);
        }
        htimer.start();
        unsigned long long schain = chain;
        if (chain != 0) {
          while ((schain >> 56) == 0) schain <<= 8;
        }
        h_decode(schain, hencoded, hdecoded, hdecsize);
        hdectime = htimer.stop() + hpredectime;
        if (conf.speed) printf("CPU decoding time: %.6f s\n", hdectime);
        hthroughput = hencsize * 0.000000001 / hdectime;
        if (conf.speed) printf("CPU decoding memory rd throughput: %8.3f Gbytes/s %8.1f%%\n", hthroughput, 100.0 * hthroughput / hmcthroughput);
        hthroughput = insize * 0.000000001 / hdectime;
        if (conf.speed) printf("CPU decoding memory wr throughput: %8.3f Gbytes/s %8.1f%%\n", hthroughput, 100.0 * hthroughput / hmcthroughput);
        if (conf.size) printf("CPU decoded size: %d bytes\n", hdecsize);
      }

      // print experiment result and record if best
      const float hCR = (100.0 * hencsize) / insize;
      printf("compression: %6.2f%% %7.3fx  (%d bytes)\n", hCR, 100.0 / hCR, hencsize);
      if (hbestCR > hCR) {
        hbestCR = hCR;
        hbestPipe = chain;
        hbestEncSize = hencsize;
      }
#endif

      // print output
      if (conf.csv) {
#ifdef USE_GPU
        fprintf(fres, " %d", dencsize);
#else
  #ifdef USE_CPU
        fprintf(fres, " %d", hencsize);
  #endif
#endif
        if (conf.decom) {
#ifdef USE_CPU
          fprintf(fres, ", %f, %f", henctime, hdectime);
#endif
#ifdef USE_GPU
          fprintf(fres, ", %f, %f", denctime, ddectime);
#endif
        }
        fprintf(fres, "\n");
      }
#ifdef USE_GPU
  #ifdef USE_CPU
      data.push_back(Elem{chain, 1.0f * insize / hencsize, 1.0f * insize / henctime, 1.0f * insize / hdectime, 1.0f * insize / denctime, 1.0f * insize / ddectime});
  #else
    #ifdef USE_GPU
      data.push_back(Elem{chain, 1.0f * insize / dencsize, -1.0f, -1.0f, 1.0f * insize / denctime, 1.0f * insize / ddectime});
    #endif
  #endif
#else
  #ifdef USE_CPU
      data.push_back(Elem{chain, 1.0f * insize / hencsize, 1.0f * insize / henctime, 1.0f * insize / hdectime, -1.0f, -1.0f});
  #endif
#endif

#if defined(USE_CPU) || defined(USE_GPU)
      if ((algorithms == 1) && (strcmp(argv[2], "AL") == 0)) {
  #ifndef USE_CPU
        const int hencsize = dencsize;  // hack
        byte* const hencoded = new byte [dencsize];  // hack
        cudaMemcpy(hencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);  // hack
  #endif
        printf("\n");
        const double ientropy_1 = Entropy((byte*)input, insize / sizeof(byte));
        printf("input byte entropy:   %6.3f bits  (%.2fx)\n", ientropy_1, 8.0 / ientropy_1);
        const double oentropy_1 = Entropy((byte*)hencoded, hencsize / sizeof(byte));
        printf("output byte entropy:  %6.3f bits  (%.2fx)\n\n", oentropy_1, 8.0 / oentropy_1);

        const double ientropy_2 = Entropy((unsigned short*)input, insize / sizeof(unsigned short));
        printf("input short entropy:  %6.3f bits  (%.2fx)\n", ientropy_2, 16.0 / ientropy_2);
        const double oentropy_2 = Entropy((unsigned short*)hencoded, hencsize / sizeof(unsigned short));
        printf("output short entropy: %6.3f bits  (%.2fx)\n\n", oentropy_2, 16.0 / oentropy_2);

        const double ientropy_4 = entropy((unsigned int*)input, insize / sizeof(unsigned int));
        printf("input int entropy:    %6.3f bits  (%.2fx)\n", ientropy_4, 32.0 / ientropy_4);
        const double oentropy_4 = entropy((unsigned int*)hencoded, hencsize / sizeof(unsigned int));
        printf("output int entropy:   %6.3f bits  (%.2fx)\n\n", oentropy_4, 32.0 / oentropy_4);

        const double ientropy_8 = entropy((unsigned long long*)input, insize / sizeof(unsigned long long));
        printf("input long entropy:   %6.3f bits  (%.2fx)\n", ientropy_8, 64.0 / ientropy_8);
        const double oentropy_8 = entropy((unsigned long long*)hencoded, hencsize / sizeof(unsigned long long));
        printf("output long entropy:  %6.3f bits  (%.2fx)\n\n", oentropy_8, 64.0 / oentropy_8);

        printf("output byte frequency\n");
        Frequency((byte*)hencoded, hencsize / sizeof(byte));
        printf("\n");

        printf("output short frequency\n");
        Frequency((unsigned short*)hencoded, hencsize / sizeof(unsigned short));
        printf("\n");

        printf("output int frequency\n");
        frequency((unsigned int*)hencoded, hencsize / sizeof(unsigned int));
        printf("\n");

        printf("output long frequency\n");
        frequency((unsigned long long*)hencoded, hencsize / sizeof(unsigned long long));
        printf("\n");
  #ifndef USE_CPU
        delete [] hencoded;  // hack
  #endif
      }
#endif

      // verify results
      bool bug = false;
      if (conf.verify && conf.decom) {
#ifdef USE_GPU
        if (ddecsize != dpreencsize) {fprintf(stderr, "ERROR: ddecode size wrong: is %d instead of %d\n\n", ddecsize, dpreencsize); bug = true;}
        const int size = std::min(dpreencsize, ddecsize);
        unsigned int* d_min_loc;
        cudaMalloc(&d_min_loc, sizeof(unsigned int));
        cudaMemset(d_min_loc, -1, sizeof(unsigned int));
        dcompareData<<<(size + TPB - 1) / TPB, TPB>>>(size, d_decoded, d_preencdata, d_min_loc);
        unsigned int dmin_loc;
        cudaMemcpy(&dmin_loc, d_min_loc, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(d_min_loc);
        if (dmin_loc < size) {
          byte data, corr;
          cudaMemcpy(&data, d_decoded + dmin_loc, sizeof(byte), cudaMemcpyDeviceToHost);
          cudaMemcpy(&corr, d_preencdata + dmin_loc, sizeof(byte), cudaMemcpyDeviceToHost);
          fprintf(stderr, "ERROR: GPU decoded result wrong at pos %d: is %x instead of %x\n\n", dmin_loc, data, corr);
          bug = true;
        }
#endif

#ifdef USE_CPU
        if (hdecsize != hpreencsize) {fprintf(stderr, "ERROR: hdecode size wrong: is %d instead of %d\n\n", hdecsize, hpreencsize); bug = true;}
        for (int i = 0; i < std::min(hpreencsize, hdecsize); i++) {
          if (hdecoded[i] != hpreencdata[i]) {fprintf(stderr, "ERROR: CPU decoded result wrong at pos %d: is %x instead of %x\n\n", i, hdecoded[i], hpreencdata[i]); bug = true; break;}
        }
#endif
      }

#ifdef USE_GPU
#ifdef USE_CPU
      if (conf.verify) {
        byte* const dencoded = new byte [dencsize];
        cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
        if (hencsize != dencsize) {fprintf(stderr, "ERROR: hencsize and dencsize differ: %d vs %d\n\n", hencsize, dencsize); bug = true;}
        for (int i = 0; i < std::min(hencsize, dencsize); i++) {
          if (hencoded[i] != dencoded[i]) {fprintf(stderr, "ERROR: CPU and GPU encoded results differ at pos %d: %x vs %x\n\n", i, hencoded[i], dencoded[i]); bug = true; break;}
        }
        delete [] dencoded;
      }
#endif
#endif

      if (conf.verify) {
#ifdef USE_CPU
        if (!bug && (algorithms == 1)) {
          //FILE* const f = fopen("LC.encoded", "wb");  assert(f != NULL);
          //const int num = fwrite(hencoded, sizeof(byte), hencsize, f);  assert(num == hencsize);
          //fclose(f);
        }
#else
        if (!bug && (algorithms == 1)) {
          //byte* const dencoded = new byte [dencsize];
          //cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
          //FILE* const f = fopen("LC.encoded", "wb");  assert(f != NULL);
          //const int num = fwrite(dencoded, sizeof(byte), dencsize, f);  assert(num == dencsize);
          //fclose(f);
          //delete [] dencoded;
        }
#endif

        if (bug) {
          exit(-1);
        } else {
          printf("verification passed\n");
        }
      }
/*
#ifdef USE_CPU
        if (algorithms == 1) {
          char fn [256];
          sprintf(fn, "%s.lc", argv[1]);
          FILE* const f = fopen(fn, "wb");  assert(f != NULL);
          const int num = fwrite(hencoded, sizeof(byte), hencsize, f);  assert(num == hencsize);
          fclose(f);
        }
#endif
*/
      printf("\n");  // end of iteration

      carrypos = 0;
      do {
        combin += 1ULL << (carrypos * 8);
        if (((combin >> (carrypos * 8)) & 0xff) < comp_list[carrypos].size()) break;
        combin &= ~(0xffULL << (carrypos * 8));
        carrypos++;
      } while (carrypos < stages);
    } while (carrypos < stages);
  }

  // print result summary
  if (conf.csv) fclose(fres);
  if (algorithms > 1) {
#ifdef USE_GPU
    int dbestLen = 0;
    unsigned short* const dbestSize = new unsigned short [dchunks];
    cudaMemcpy(dbestSize, d_bestSize, dchunks * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    for (int i = 0; i < dchunks; i++) dbestLen += dbestSize[i];
    delete [] dbestSize;
    dbestLen += 6 + 2 * dchunks + 4 * dchunks;  // 6-byte header, 2 bytes to store size, assume 4 bytes to store algorithm
    const float dbestPC = (100.0 * dbestLen) / insize;

    printf("Overall best\n------------\n");
    printf("preprocs: %s\n", getPreprocessors(prepros).c_str());
    printf("pipeline: %s\n", getPipeline(dbestPipe, stages).c_str());
    printf("compression: %6.2f%% %7.3fx  (%d bytes)\n", dbestCR, 100.0 / dbestCR, dbestEncSize);
    printf("per chunk:   %6.2f%% %7.3fx\n\n", dbestPC, 100.0 / dbestPC);
#endif

#ifdef USE_CPU
    int hbestLen = 0;
    for (int i = 0; i < hchunks; i++) hbestLen += hbestSize[i];
    hbestLen += 6 + 2 * hchunks + 4 * hchunks;  // 6-byte header, 2 bytes to store size, assume 4 bytes to store algorithm
    const float hbestPC = (100.0 * hbestLen) / insize;

    printf("Overall best\n------------\n");
    printf("preprocs: %s\n", getPreprocessors(prepros).c_str());
    printf("pipeline: %s\n", getPipeline(hbestPipe, stages).c_str());
    printf("compression: %6.2f%% %7.3fx  (%d bytes)\n", hbestCR, 100.0 / hbestCR, hbestEncSize);
    printf("per chunk:   %6.2f%% %7.3fx\n\n", hbestPC, 100.0 / hbestPC);
#endif

    if (conf.decom) {
#ifdef USE_GPU
      printf("Pareto device encoding\n----------------------    *                       *\n");
      std::sort(data.begin(), data.end(), compareElemDencThru);
      for (int i = 0; i < data.size(); i++) {
        bool ok = true;
        if (data[i].CR <= 1.0f) {
          ok = false;
        } else {
          for (int j = i + 1; j < data.size(); j++) {
            if (data[i].DencThru < data[j].DencThru) {
              ok = false;
              break;
            }
          }
        }
        if (ok) {
          printf("%-21s %7.3fx %7.3f %7.3f", getPipeline(data[i].pipe, stages).c_str(), data[i].CR, data[i].HencThru / 1000000000.0f, data[i].HdecThru / 1000000000.0f);
          printf(" %7.3f %7.3f\n", data[i].DencThru / 1000000000.0f, data[i].DdecThru / 1000000000.0f);
        }
      }
      printf("\n");

      printf("Pareto device decoding\n----------------------    *                               *\n");
      std::sort(data.begin(), data.end(), compareElemDdecThru);
      for (int i = 0; i < data.size(); i++) {
        bool ok = true;
        if (data[i].CR <= 1.0f) {
          ok = false;
        } else {
          for (int j = i + 1; j < data.size(); j++) {
            if (data[i].DdecThru < data[j].DdecThru) {
              ok = false;
              break;
            }
          }
        }
        if (ok) {
          printf("%-21s %7.3fx %7.3f %7.3f", getPipeline(data[i].pipe, stages).c_str(), data[i].CR, data[i].HencThru / 1000000000.0f, data[i].HdecThru / 1000000000.0f);
          printf(" %7.3f %7.3f\n", data[i].DencThru / 1000000000.0f, data[i].DdecThru / 1000000000.0f);
        }
      }
      printf("\n");
#endif

#ifdef USE_CPU
      printf("Pareto host encoding\n--------------------      *        *\n");
      std::sort(data.begin(), data.end(), compareElemHencThru);
      for (int i = 0; i < data.size(); i++) {
        bool ok = true;
        if (data[i].CR <= 1.0f) {
          ok = false;
        } else {
          for (int j = i + 1; j < data.size(); j++) {
            if (data[i].HencThru < data[j].HencThru) {
              ok = false;
              break;
            }
          }
        }
        if (ok) {
          printf("%-21s %7.3fx %7.3f %7.3f", getPipeline(data[i].pipe, stages).c_str(), data[i].CR, data[i].HencThru / 1000000000.0f, data[i].HdecThru / 1000000000.0f);
          printf(" %7.3f %7.3f\n", data[i].DencThru / 1000000000.0f, data[i].DdecThru / 1000000000.0f);
        }
      }
      printf("\n");

      printf("Pareto host decoding\n--------------------      *                *\n");
      std::sort(data.begin(), data.end(), compareElemHdecThru);
      for (int i = 0; i < data.size(); i++) {
        bool ok = true;
        if (data[i].CR <= 1.0f) {
          ok = false;
        } else {
          for (int j = i + 1; j < data.size(); j++) {
            if (data[i].HdecThru < data[j].HdecThru) {
              ok = false;
              break;
            }
          }
        }
        if (ok) {
          printf("%-21s %7.3fx %7.3f %7.3f", getPipeline(data[i].pipe, stages).c_str(), data[i].CR, data[i].HencThru / 1000000000.0f, data[i].HdecThru / 1000000000.0f);
          printf(" %7.3f %7.3f\n", data[i].DencThru / 1000000000.0f, data[i].DdecThru / 1000000000.0f);
        }
      }
      printf("\n");
#endif
    }
  }

  // clean up
  delete [] input;

#ifdef USE_GPU
  cudaFree(d_input);
  cudaFree(d_preencdata);
  cudaFree(d_predecdata);
  cudaFree(d_encoded);
  cudaFree(d_encsize);
  cudaFree(d_decoded);
  cudaFree(d_decsize);
  cudaFree(d_bestSize);
  CheckCuda(__LINE__);
#endif

#ifdef USE_CPU
  delete [] hpreencdata;
  delete [] hpredecdata;
  delete [] hencoded;
  delete [] hdecoded;
  delete [] hbestSize;
#endif

  return 0;
}
