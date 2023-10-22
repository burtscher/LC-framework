/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2023, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
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

#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
/*##include-beg##*/
// inlcude files to be inserted
/*##include-end##*/


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
      if (tid < len & 3) {
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
      if (tid < rlen & 3) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}

static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
}


static inline __device__ int propagate_carry(const int value, const int chunkID, volatile int fullcarry [], volatile int partcarry [], volatile byte status [], int* const s_fullc)
{
  const int lane = threadIdx.x % WS;
  const bool lastthread = (threadIdx.x == (TPB - 1));

  if (lastthread) {
    *s_fullc = 0;
    if (chunkID == 0) {
      fullcarry[0] = value;
      __threadfence();
      status[0] = 2;
    } else {
      partcarry[chunkID] = value;
      __threadfence();
      status[chunkID] = 1;
    }
  }

  if (chunkID > 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      __syncwarp();  // optional

      const int cidm1 = chunkID - 1;
      int stat = 1;
      do {
        if (cidm1 - lane >= 0) {
          stat = status[cidm1 - lane];
        }
      } while ((__any_sync(~0, stat == 0)) || (__all_sync(~0, stat != 2)));
      __threadfence();
      const int mask = __ballot_sync(~0, stat == 2);
      const int pos = __ffs(mask) - 1;
      int partc = (lane < pos) ? partcarry[chunkID - pos + lane] : 0;
      partc += __shfl_xor_sync(~0, partc, 1);  // MB: use reduction on 8.6 devices
      partc += __shfl_xor_sync(~0, partc, 2);
      partc += __shfl_xor_sync(~0, partc, 4);
      partc += __shfl_xor_sync(~0, partc, 8);
      partc += __shfl_xor_sync(~0, partc, 16);
      if (lastthread) {
        const int fullc = partc + fullcarry[cidm1 - pos];
        fullcarry[chunkID] = fullc + value;
        __threadfence();
        status[chunkID] = 2;
        *s_fullc = fullc;
      }
    }
  }
  __syncthreads();  // wait for s_fullc to be available

  return *s_fullc;
}


static __global__ __launch_bounds__(TPB, 2)
void d_encode(const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int* const __restrict__ outsize, int* const __restrict__ fullcarry, int* const __restrict__ partcarry, byte* const __restrict__ status)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 1 - WS;

  // initialize
  const int chunks = (insize + CS - 1) / CS;  // round up
  int* const head_out = (int*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[3];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= insize) break;
    
    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];
    
    // load chunk
    const int osize = min(CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];
    __syncthreads();  // chunk produced, chunk[last] consumed

    // encode chunk
    int csize = osize;
    bool good = true;
    /*##comp-encoder-beg##*/
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_CLOG_1(csize, in, out, temp);
    }
    __syncthreads();  // chunk transformed
    /*##comp-encoder-end##*/
    
    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    const int offs = propagate_carry(csize, chunkID, fullcarry, partcarry, status, (int*)temp);
    __syncthreads();  // temp consumed

    // store chunk
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (int i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];
      __syncthreads();  // re-loading input done
    }
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunks - 1]] - output;
    }
  } while (true);
}

struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001 * ms;}
};

static void CheckCuda(const int line)  //MB: remove later
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char* argv [])
{
  /*##print-beg##*/
  /*##print-end##*/	
  printf("Copyright 2023 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s input_file_name compressed_file_name [performance_analysis (y)]\n\n", argv[0]);  exit(-1);}
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const int fsize = ftell(fin);  assert(fsize > 0);
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %d bytes\n", insize);

  // Check if the third argument is "y" to enable performance analysis
  char* perf_str = argv[3];
  bool perf = false;
  if (perf_str != nullptr && strcmp(perf_str, "y") == 0) {
    perf = true;
  } else if (perf_str != nullptr && strcmp(perf_str, "y") != 0) {
    printf("Invalid performance analysis argument. Use 'y' or leave it empty.\n");
    exit(-1);
  }

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  const int chunks = (insize + CS - 1) / CS;  // round up
  CheckCuda(__LINE__);
  const int maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS;
  
  // allocate GPU memory
  byte* dencoded;
  cudaMallocHost((void **)&dencoded, maxsize);
  byte* d_input;
  cudaMalloc((void **)&d_input, insize);
  cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  int* d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(int));
  CheckCuda(__LINE__);

  byte* dpreencdata;
  cudaMalloc((void **)&dpreencdata, insize);
  cudaMemcpy(dpreencdata, d_input, insize, cudaMemcpyDeviceToDevice);
  int dpreencsize = insize;
  
  if (perf) {
    // warm up
    int* d_fullcarry_dummy;
    int* d_partcarry_dummy;
    byte* d_status_dummy;
    cudaMalloc((void **)&d_fullcarry_dummy, chunks * sizeof(int));
    cudaMalloc((void **)&d_partcarry_dummy, chunks * sizeof(int));
    cudaMalloc((void **)&d_status_dummy, chunks * sizeof(byte));
    d_reset<<<1, 1>>>();
    cudaMemset(d_status_dummy, 0, chunks * sizeof(byte));
    d_encode<<<blocks, TPB>>>(dpreencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry_dummy, d_partcarry_dummy, d_status_dummy);
    cudaFree(d_fullcarry_dummy);
    cudaFree(d_partcarry_dummy);
    cudaFree(d_status_dummy);   
  }

  GPUTimer dtimer;
  dtimer.start();
  /*##pre-encoder-beg##*/
  /*##pre-encoder-end##*/
  int* d_fullcarry;
  int* d_partcarry;
  byte* d_status;
  cudaMalloc((void **)&d_fullcarry, chunks * sizeof(int));
  cudaMalloc((void **)&d_partcarry, chunks * sizeof(int));
  cudaMalloc((void **)&d_status, chunks * sizeof(byte));
  d_reset<<<1, 1>>>();
  cudaMemset(d_status, 0, chunks * sizeof(byte));
  d_encode<<<blocks, TPB>>>(dpreencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry, d_partcarry, d_status);
  cudaFree(d_fullcarry);
  cudaFree(d_partcarry);
  cudaFree(d_status);
  cudaDeviceSynchronize();
  double runtime = dtimer.stop();
  
  // get encoded GPU result
  int dencsize = 0;
  cudaMemcpy(&dencsize, d_encsize, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
  printf("encoded size: %d bytes\n", dencsize);
  CheckCuda(__LINE__);

  const float CR = (100.0 * dencsize) / insize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  if (perf) {
    printf("encoding time: %.6f s\n", runtime);
    double throughput = insize * 0.000000001 / runtime;
    printf("encoding throughput: %8.3f Gbytes/s\n", throughput);
    CheckCuda(__LINE__);
  }

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(dencoded, 1, dencsize, fout);
  fclose(fout);

  // clean up GPU memory
  cudaFree(d_input);
  cudaFree(d_encoded);
  cudaFree(d_encsize);
  CheckCuda(__LINE__);

  // clean up
  cudaFreeHost(input);
  cudaFreeHost(dencoded);
  return 0;
}
