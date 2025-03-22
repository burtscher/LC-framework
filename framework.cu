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


//#define NDEBUG


#include "lc.h"


int main(int argc, char* argv [])
{
  printf("LC Compression Framework v1.2 (%s)\n", __FILE__);

#ifndef USE_GPU
  #ifndef USE_CPU
  fprintf(stderr, "ERROR: must define 'USE_CPU', 'USE_GPU', or both when compiling code\n\n");
  throw std::runtime_error("LC error");
  #else
  printf("CPU version\n");
  #endif
#else
  #ifndef USE_CPU
    #if defined(__AMDGCN_WAVEFRONT_SIZE)
    printf("AMD ");
    #endif
  printf("GPU version\n");
  #else
  printf("Combined CPU + GPU version\n");
  #endif
#endif

  printf("Copyright 2024 Texas State University\n\n");

  // perform system checks
  if (CS % 8 != 0) {fprintf(stderr, "ERROR: CS must be a multiple of 8\n\n"); throw std::runtime_error("LC error");}
  const int endian = 1;
  if (*((char*)(&endian)) != 1) {fprintf(stderr, "ERROR: framework only supports little-endian systems\n\n"); throw std::runtime_error("LC error");}
  if (sizeof(long long) != 8) {fprintf(stderr, "ERROR: long long must be 8 bytes\n\n"); throw std::runtime_error("LC error");}
  if (sizeof(int) != 4) {fprintf(stderr, "ERROR: int must be 4 bytes\n\n"); throw std::runtime_error("LC error");}
  if (sizeof(short) != 2) {fprintf(stderr, "ERROR: short must be 2 bytes\n\n"); throw std::runtime_error("LC error");}
  if (sizeof(char) != 1) {fprintf(stderr, "ERROR: char must be 1 byte\n\n"); throw std::runtime_error("LC error");}

  // print usage message if needed
  if ((argc < 3) || (argc > 6) || (argc == 4)) {
    printUsage(argv);
    return -1;
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
    if (argc != 3) {printUsage(argv); return -1;}
    char regex [] = {'.', '+', ' ', '.', '+', 0};
    comp_list = getStages(comp_name2num, regex, stages, algorithms);
    conf = {false, false, false, false, true, true, false};  // -speed, -size, -warmup, -memcopy, +decom, +verify, -csv
  } else if (strcmp(argv[2], "CR") == 0) {  // exhaustive with only CR, no speed
    if (argc != 5) {printUsage(argv); return -1;}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (algorithms < 1) {fprintf(stderr, "ERROR: need at least one algorithm\n\n"); throw std::runtime_error("LC error");}
    conf = {false, false, false, false, false, false, true};  // -speed, -size, -warmup, -memcopy, -decom, -verify, +csv
    ext = ".CR" + std::to_string(stages);
  } else if (strcmp(argv[2], "EX") == 0) {  // exhaustive with all metrics
    if ((argc != 5) && (argc != 6)) {printUsage(argv); return -1;}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (argc == 6) verifs = getItems(verif_name2num, argv[5]);
    if (algorithms < 1) {fprintf(stderr, "ERROR: need at least one algorithm\n\n"); throw std::runtime_error("LC error");}
    conf = {false, false, true, false, true, true, true};  // -speed, -size, +warmup, -memcopy, +decom, +verify, +csv
    ext = ".EX" + std::to_string(stages);
  } else if (strcmp(argv[2], "PR") == 0) {  // just print stages and exit
    if ((argc != 5) && (argc != 6)) {printUsage(argv); return -1;}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (argc == 6) verifs = getItems(verif_name2num, argv[5]);
    printStages(prepros, prepro_name2num, comp_list, comp_name2num, stages, algorithms);
    return 0;
  } else if (strcmp(argv[2], "AL") == 0) {  // single algorithm with all metrics
    if ((argc != 5) && (argc != 6)) {printUsage(argv); return -1;}
    prepros = getItems(prepro_name2num, argv[3]);
    comp_list = getStages(comp_name2num, argv[4], stages, algorithms);
    if (argc == 6) verifs = getItems(verif_name2num, argv[5]);
    if (algorithms != 1) {fprintf(stderr, "ERROR: pipeline must describe one algorithm\n\n"); throw std::runtime_error("LC error");}
    conf = {true, true, true, true, true, true, false};  // +speed, +size, +warmup, +memcopy, +decom, +verify, -csv
  } else {  // unknown mode
    printUsage(argv);
    return -1;
  }
  printStages(prepros, prepro_name2num, comp_list, comp_name2num, stages, algorithms);

  // read input file
  printf("input: %s\n", argv[1]);
  FILE* const fin = fopen(argv[1], "rb"); assert(fin != NULL);
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); throw std::runtime_error("LC error");}
  if (fsize >= 9223372036854775807) {fprintf(stderr, "ERROR: input file too large\n\n"); throw std::runtime_error("LC error");}
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(input, 1, fsize, fin); assert(insize == fsize);
  fclose(fin);
  printf("input size: %lld bytes\n\n", insize);

#ifdef USE_GPU
  // copy input to GPU
  byte* d_input;
  cudaMalloc((void **)&d_input, insize);
  cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);

  // get and print GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); throw std::runtime_error("LC error");}
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
    long long dpreencsize = insize;
    d_preprocess_encode(dpreencsize, d_preencdata, prepros);
    cudaFree(d_preencdata);
  }
  cudaMalloc((void **)&d_preencdata, insize);
  cudaMemcpy(d_preencdata, d_input, insize, cudaMemcpyDeviceToDevice);
  long long dpreencsize = insize;
  GPUTimer dtimer;
  dtimer.start();
  d_preprocess_encode(dpreencsize, d_preencdata, prepros);
  const double dpreenctime = (prepros.size() == 0) ? 0 : dtimer.stop();
  if (conf.speed) printf("GPU preprocessor encoding time: %.6f s\n", dpreenctime);
  if (conf.size) printf("GPU preprocessor encoded size: %lld bytes\n\n", dpreencsize);

  // time GPU preprocessor decoding
  long long dpredecsize = 0;
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
    if (conf.size) printf("GPU preprocessor decoded size: %lld bytes\n\n", dpredecsize);
  }
#endif

#ifdef USE_CPU
  // time CPU preprocessor encoding
  if (conf.warmup) {
    byte* hpreencdata = new byte [insize];
    std::copy(input, input + insize, hpreencdata);
    long long hpreencsize = insize;
    h_preprocess_encode(hpreencsize, hpreencdata, prepros);
    delete [] hpreencdata;
  }
  byte* hpreencdata = new byte [insize];
  std::copy(input, input + insize, hpreencdata);
  long long hpreencsize = insize;
  CPUTimer htimer;
  htimer.start();
  h_preprocess_encode(hpreencsize, hpreencdata, prepros);
  const double hpreenctime = (prepros.size() == 0) ? 0 : htimer.stop();
  if (conf.speed) printf("CPU preprocessor encoding time: %.6f s\n", hpreenctime);
  if (conf.size) printf("CPU preprocessor encoded size: %lld bytes\n\n", hpreencsize);

  // time CPU preprocessor decoding
  long long hpredecsize = 0;
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
    if (conf.size) printf("CPU preprocessor decoded size: %lld bytes\n\n", hpredecsize);
  }
#endif

  // verify results
  if (conf.verify && conf.decom) {
    bool bug = false;

#ifdef USE_GPU
    if (dpredecsize != insize) {fprintf(stderr, "ERROR: dpredec size wrong: is %lld instead of %lld\n\n", dpredecsize, insize); bug = true;}
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
    if (hpredecsize != insize) {fprintf(stderr, "ERROR: hpredec size wrong: is %lld instead of %lld\n\n", hpredecsize, insize); bug = true;}
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
    if (dpreencsize != hpreencsize) {fprintf(stderr, "ERROR: hpreencsize and dpreencsize differ: %lld vs %lld\n\n", hpreencsize, dpreencsize); bug = true;}
    for (long long i = 0; i < std::min(hpreencsize, dpreencsize); i++) {
      if (dpreencdata[i] != hpreencdata[i]) {
        fprintf(stderr, "ERROR: CPU and GPU preprocessor encoded results differ at pos %lld: %x vs %x\n\n", i, hpreencdata[i], dpreencdata[i]);
        bug = true;
        break;
      }
    }
    delete [] dpreencdata;
    printf("compared preenc CPU to GPU\n");  //MB: remove later

    dpredecdata = new byte [dpredecsize];
    cudaMemcpy(dpredecdata, d_predecdata, dpredecsize, cudaMemcpyDeviceToHost);
    if (dpredecsize != hpredecsize) {fprintf(stderr, "ERROR: hpredecsize and dpredecsize differ: %lld vs %lld\n\n", hpredecsize, dpredecsize); bug = true;}
    for (long long i = 0; i < std::min(hpredecsize, dpredecsize); i++) {
      if (dpredecdata[i] != hpredecdata[i]) {
        fprintf(stderr, "ERROR: CPU and GPU preprocessor decoded results differ at pos %lld: %x vs %x\n\n", i, hpredecdata[i], dpredecdata[i]);
        bug = true;
        break;
      }
    }
    delete [] dpredecdata;
    printf("compared predec CPU to GPU\n");  //MB: remove later
#endif
#endif

    if (bug) {
      throw std::runtime_error("LC error");
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
    fprintf(fres, "size [bytes], %lld\n", insize);
    time_t curtime;
    time(&curtime);
    fprintf(fres, "date/time, %s", ctime(&curtime));
    //char name [256];
    //gethostname(name, sizeof(name));
    fprintf(fres, "host name, %s\n", "unknown");

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
  const long long dchunks = (dpreencsize + CS - 1) / CS;  // round up
  const long long dmaxsize = 2 * sizeof(long long) + dchunks * sizeof(short) + dchunks * CS;  //MB: adjust later
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, dmaxsize);
  long long* d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(long long));
  byte* d_decoded;
  cudaMalloc((void **)&d_decoded, dpreencsize);
  long long* d_decsize;
  cudaMalloc((void **)&d_decsize, sizeof(long long));
  CheckCuda(__LINE__);
#endif

#ifdef USE_CPU
  // allocate CPU memory
  const long long hchunks = (hpreencsize + CS - 1) / CS;  // round up
  const long long hmaxsize = 2 * sizeof(long long) + hchunks * sizeof(short) + hchunks * CS;  //MB: adjust later
  byte* const hencoded = new byte [hmaxsize];
  long long hencsize = 0;
  byte* const hdecoded = new byte [hpreencsize];
  long long hdecsize = 0;
#endif

#ifdef USE_GPU
  // run GPU experiments
  float dbestCR = 100.0;
  unsigned long long dbestPipe = 0;
  long long dbestEncSize = insize;
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
  long long hbestEncSize = insize;
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
        long long* d_fullcarry;
        cudaMalloc((void **)&d_fullcarry, dchunks * sizeof(long long));
        d_reset<<<1, 1>>>();
        cudaMemset(d_fullcarry, 0, dchunks * sizeof(long long));
        d_encode<<<blocks, TPB>>>(chain, d_preencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry);
        cudaFree(d_fullcarry);
        cudaDeviceSynchronize();
        CheckCuda(__LINE__);
      }
      GPUTimer dtimer;
      dtimer.start();
      long long* d_fullcarry;
      cudaMalloc((void **)&d_fullcarry, dchunks * sizeof(long long));
      d_reset<<<1, 1>>>();
      cudaMemset(d_fullcarry, 0, dchunks * sizeof(long long));
      d_encode<<<blocks, TPB>>>(chain, d_preencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry);
      cudaFree(d_fullcarry);
      cudaDeviceSynchronize();
      denctime = dtimer.stop() + dpreenctime;
      if (conf.speed) printf("GPU encoding time: %.6f s\n", denctime);
      double dthroughput = insize * 0.000000001 / denctime;
      if (conf.speed) printf("GPU encoding memory rd throughput: %8.3f Gbytes/s %8.1f%%\n", dthroughput, 100.0 * dthroughput / dmcthroughput);
      CheckCuda(__LINE__);

      // get encoded GPU result
      long long dencsize = 0;
      cudaMemcpy(&dencsize, d_encsize, sizeof(long long), cudaMemcpyDeviceToHost);
      dthroughput = dencsize * 0.000000001 / denctime;
      if (conf.speed) printf("GPU encoding memory wr throughput: %8.3f Gbytes/s %8.1f%%\n", dthroughput, 100.0 * dthroughput / dmcthroughput);
      if (conf.size) printf("GPU encoded size: %lld bytes\n", dencsize);
      CheckCuda(__LINE__);

      dbestChunkSize<<<1, TPB>>>(d_encoded, d_bestSize);

      // time GPU decoding
      long long ddecsize = 0;
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
        cudaMemcpy(&ddecsize, d_decsize, sizeof(long long), cudaMemcpyDeviceToHost);
        if (conf.size) printf("GPU decoded size: %lld bytes\n", ddecsize);
        CheckCuda(__LINE__);
      }

      // print experiment result and record if best
      const float dCR = (100.0 * dencsize) / insize;
      printf("compression: %6.2f%% %7.3fx  (%lld bytes)\n", dCR, 100.0 / dCR, dencsize);
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
      if (conf.size) printf("CPU encoded size: %lld bytes\n", hencsize);

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
        if (conf.size) printf("CPU decoded size: %lld bytes\n", hdecsize);
      }

      // print experiment result and record if best
      const float hCR = (100.0 * hencsize) / insize;
      printf("compression: %6.2f%% %7.3fx  (%lld bytes)\n", hCR, 100.0 / hCR, hencsize);
      if (hbestCR > hCR) {
        hbestCR = hCR;
        hbestPipe = chain;
        hbestEncSize = hencsize;
      }
#endif

      // print output
      if (conf.csv) {
#ifdef USE_GPU
        fprintf(fres, " %lld", dencsize);
#else
  #ifdef USE_CPU
        fprintf(fres, " %lld", hencsize);
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
        const long long hencsize = dencsize;  // hack
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
        if (ddecsize != dpreencsize) {fprintf(stderr, "ERROR: ddecode size wrong: is %lld instead of %lld\n\n", ddecsize, dpreencsize); bug = true;}
        const long long size = std::min(dpreencsize, ddecsize);
        unsigned long long* d_min_loc;
        cudaMalloc(&d_min_loc, sizeof(unsigned long long));
        cudaMemset(d_min_loc, -1, sizeof(unsigned long long));
        dcompareData<<<(size + TPB - 1) / TPB, TPB>>>(size, d_decoded, d_preencdata, d_min_loc);
        unsigned long long dmin_loc;
        cudaMemcpy(&dmin_loc, d_min_loc, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaFree(d_min_loc);
        if (dmin_loc < size) {
          byte data, corr;
          cudaMemcpy(&data, d_decoded + dmin_loc, sizeof(byte), cudaMemcpyDeviceToHost);
          cudaMemcpy(&corr, d_preencdata + dmin_loc, sizeof(byte), cudaMemcpyDeviceToHost);
          fprintf(stderr, "ERROR: GPU decoded result wrong at pos %lld: is %x instead of %x\n\n", dmin_loc, data, corr);
          bug = true;
        }
#endif

#ifdef USE_CPU
        if (hdecsize != hpreencsize) {fprintf(stderr, "ERROR: hdecode size wrong: is %lld instead of %lld\n\n", hdecsize, hpreencsize); bug = true;}
        for (long long i = 0; i < std::min(hpreencsize, hdecsize); i++) {
          if (hdecoded[i] != hpreencdata[i]) {fprintf(stderr, "ERROR: CPU decoded result wrong at pos %lld: is %x instead of %x\n\n", i, hdecoded[i], hpreencdata[i]); bug = true; break;}
        }
#endif
      }

#ifdef USE_GPU
#ifdef USE_CPU
      if (conf.verify) {
        byte* const dencoded = new byte [dencsize];
        cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
        if (hencsize != dencsize) {fprintf(stderr, "ERROR: hencsize and dencsize differ: %lld vs %lld\n\n", hencsize, dencsize); bug = true;}
        for (long long i = 0; i < std::min(hencsize, dencsize); i++) {
          if (hencoded[i] != dencoded[i]) {fprintf(stderr, "ERROR: CPU and GPU encoded results differ at pos %lld: %x vs %x\n\n", i, hencoded[i], dencoded[i]); bug = true; break;}
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
          throw std::runtime_error("LC error");
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
    for (long long i = 0; i < dchunks; i++) dbestLen += dbestSize[i];
    delete [] dbestSize;
    dbestLen += 6 + 2 * dchunks + 4 * dchunks;  // 6-byte header, 2 bytes to store size, assume 4 bytes to store algorithm
    const float dbestPC = (100.0 * dbestLen) / insize;

    printf("Overall best\n------------\n");
    printf("preprocs: %s\n", getPreprocessors(prepros).c_str());
    printf("pipeline: %s\n", getPipeline(dbestPipe, stages).c_str());
    printf("compression: %6.2f%% %7.3fx  (%lld bytes)\n", dbestCR, 100.0 / dbestCR, dbestEncSize);
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
    printf("compression: %6.2f%% %7.3fx  (%lld bytes)\n", hbestCR, 100.0 / hbestCR, hbestEncSize);
    printf("per chunk:   %6.2f%% %7.3fx\n\n", hbestPC, 100.0 / hbestPC);
#endif

    if (conf.decom) {
#ifdef USE_GPU
      printf("Pareto device encoding\n----------------------    *                       *\n");
      std::sort(data.begin(), data.end(), compareElemDencThru);
      for (long long i = 0; i < data.size(); i++) {
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
      for (long long i = 0; i < data.size(); i++) {
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
