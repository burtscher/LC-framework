#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>


typedef float type;  // type of a word

static const int TPB = 512;  // threads per block

static const type missing = nanf("");


static __device__ inline bool isMissing(const type x)
{
  return isnan(x);
}


static __device__ inline void reconstruct(type& a, type& b)
{
  if (isMissing(b)) {
    b = a;  // replace missing value by average
  } else {
    a = 2 * a - b;  // replace by 2 * average minus given value
  }
}


static __global__ void unshuffle(const int iter, const int step, const int shift1, const int len, const type* const __restrict__ input, type* const __restrict__ output)
{
  const int i = threadIdx.x + blockIdx.x * TPB;
  if (i < iter) {
    const int ishift = i << step;
    const int offs = ishift + shift1;
    output[offs] = input[len + i];
  }
  if ((len == 1) && (i == 0)) {
    output[0] = input[0];
  }
}


static __global__ void decode(const int iter, const int step, const int shift1, type* const __restrict__ output)
{
  const int i = threadIdx.x + blockIdx.x * TPB;
  if (i < iter) {
    const int ishift = i << step;
    const int offs = ishift + shift1;
    reconstruct(output[ishift], output[offs]);
  }
}


static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001 * ms;}
};


int main(int argc, char* argv[])
{
  printf("Progressive decoder v0.1 CUDA (%s)\n", __FILE__);
  if (argc != 3) {printf("USAGE: %s input_file_name output_file_name\n", argv[0]);  exit(-1);}

  // read input from file
  FILE* const fin = fopen(argv[1], "rb");  assert(fin != NULL);
  fseek(fin, 0, SEEK_END);  assert(ftell(fin) >= sizeof(int) + sizeof(type));
  fseek(fin, 0, SEEK_SET);
  int size;
  fread(&size, sizeof(int), 1, fin);
  printf("size: %d words, %d bytes\n", size, size * sizeof(type));
  type* const input = new type [size];
  const int insize = fread(input, sizeof(type), size, fin);
  fclose(fin);
  printf("input size: %d words, %d bytes\n", insize, insize * sizeof(type));
  for (int i = insize; i < size; i++) input[i] = missing;
  if (size < 1) {fprintf(stderr, "ERROR: input must contain at least one value\n\n");  exit(-1);}

  // check GPU
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("gpu: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  // alloc GPU memory
  type* d_input;
  cudaMalloc((void **)&d_input, size * sizeof(type));
  cudaMemcpy(d_input, input, size * sizeof(type), cudaMemcpyHostToDevice);
  type* d_output;
  cudaMalloc((void **)&d_output, size * sizeof(type));
  CheckCuda();

  // start timer
  GPUTimer timer;
  timer.start();

  // compute sizes
  int s [60];
  int k [60];
  int level = 0;
  int len = size;
  do {
    k[level] = len / 2;
    len = (len + 1) / 2;
    s[level] = len;
    level++;
  } while (len > 1);

  // launch kernels
  for (int lvl = level - 1; lvl >= 0; lvl--) {
    const int len = s[lvl];
    const int iter = k[lvl];
    const int shift1 = 1 << lvl;
    const int step = lvl + 1;
    unshuffle<<<(iter + TPB - 1) / TPB, TPB>>>(iter, step, shift1, len, d_input, d_output);
  }
  for (int lvl = level - 1; lvl >= 0; lvl--) {
    const int iter = k[lvl];
    const int shift1 = 1 << lvl;
    const int step = lvl + 1;
    decode<<<(iter + TPB - 1) / TPB, TPB>>>(iter, step, shift1, d_output);
  }

  // stop timer
  cudaDeviceSynchronize();
  double runtime = timer.stop();
  CheckCuda();
  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.3f Gbytes/s\n", size * sizeof(type) * 0.000000001 / runtime);

  // get result
  type* const output = new type [size];
  cudaMemcpy(output, d_output, size * sizeof(type), cudaMemcpyDeviceToHost);
  CheckCuda();

  // write output to file
  FILE* const fout = fopen(argv[2], "wb");  assert(fout != NULL);
  fwrite(output, sizeof(type), size, fout);
  fclose(fout);

  // clean up
  delete [] input;
  delete [] output;
  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}
