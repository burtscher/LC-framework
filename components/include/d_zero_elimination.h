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


#ifndef zero_elimination_device
#define zero_elimination_device


//special case for byte and short data
template <typename T, bool check = false>
static __device__ inline bool d_ZEencodebyteshort(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  using type = T;
  using ull = unsigned long long;
  const int bitsperword = 8 * sizeof(type);
  const int bitsperlong = 8 * sizeof(ull);
  const int wordsperlong = bitsperlong / bitsperword;
  const int bytesperthread = CS / TPB;
  const ull* const in_l = (ull*)in;
  const int csize = insize * sizeof(T);
  assert(bytesperthread % sizeof(ull) == 0);
  assert(bytesperthread / sizeof(type) <= sizeof(int) * 8);
  assert(bytesperthread / sizeof(ull) * wordsperlong >= 8);
  assert(std::is_unsigned<type>::value);

  // output bitmaps and count non-zero values
  const int tid = threadIdx.x;
  int bmp = 0, cnt = 0;
  if (tid * bytesperthread < csize) {
    for (int i = 0; i < bytesperthread / sizeof(ull); i++) {
      const ull lval = in_l[tid * (bytesperthread / sizeof(ull)) + i];
      int bm = 0;
      for (int j = 0; j < wordsperlong; j++) {
        const type val = lval >> (j * bitsperword);
        bm |= (val != 0) << j;
      }
      bmp |= bm << (i * wordsperlong);
    }
    if (tid * bytesperthread - (csize - bytesperthread) > 0) {
      bmp &= ~(-1 << ((csize % bytesperthread + sizeof(type) - 1) / sizeof(type)));
    }
    //if constexpr (sizeof(type) == 1) ((int*)bmout)[tid] = bmp;
    if constexpr (sizeof(type) == 1) {
      bmout[tid * 4] = bmp;
      bmout[tid * 4 + 1] = bmp >> 8;
      bmout[tid * 4 + 2] = bmp >> 16;
      bmout[tid * 4 + 3] = bmp >> 24;
    }
    if constexpr (sizeof(type) == 2) bmout[tid] = bmp;
    if constexpr (sizeof(type) == 4) ((byte*)bmout)[tid] = bmp;
    cnt = __popc(bmp);
  }

  int pos = block_prefix_sum(cnt, temp_w);
  if (tid == TPB - 1) temp_w[WS] = pos;
  if constexpr (check) {
    if (__syncthreads_or(pos > datasize)) return false;
  } else {
    __syncthreads();
  }
  pos -= cnt;

  // output non-zero values
  if (bmp != 0) {
    for (int i = 0; i < bytesperthread / sizeof(ull); i++) {
      const ull lval = in_l[tid * (bytesperthread / sizeof(ull)) + i];
      const int bm = bmp >> (i * wordsperlong);
      for (int j = 0; j < wordsperlong; j++) {
        if ((bm >> j) & 1) {
          dataout[pos++] = lval >> (j * bitsperword);
        }
      }
    }
  }

  datasize = temp_w[WS];
  return true;
}


//warp-based one word per thread
template <typename T, bool check = false>
static __device__ inline bool d_ZEencode1wordperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warps = TPB / WS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // count non-zero values and output bitmaps
  const bool active = (tid < insize);
  const T val = active ? in[tid] : 0;
  const bool havenonzeroval = (active && (val != 0));
#if defined(WS) && (WS == 64)
  const unsigned long long tmp = __ballot_sync(~0, havenonzeroval);
  const int bm = (lane < 32) ? (int)tmp : (int)(tmp >> 32);
#else  
  const int bm = __ballot_sync(~0, havenonzeroval);
#endif  
  const int cnt = __popc(bm);
  const int subwarps = TPB / 32;
#if defined(WS) && (WS == 64)
  const int sublane = lane & 31;
  const int subwarp = threadIdx.x / 32;
  if (active && (sublane % 8 == 0)) bmout_b[tid / 8] = bm >> sublane;
#else
  const int sublane = lane;
  const int subwarp = warp;
  if (active && (lane % 8 == 0)) bmout_b[tid / 8] = bm >> lane;
#endif 
  if constexpr (sizeof(T) > 1) {  //MB: (never used) maybe somewhere else zero out last word of bitmap before a barrier and calling this function?
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  // output non-zero values
  if (havenonzeroval) {
    const int loc = temp_w[subwarp] - cnt + __popc(bm & ((1 << sublane) - 1));
    dataout[loc] = val;
  }

  datasize = temp_w[subwarps - 1];
  return true;
}


//warp-based two words per thread
template <typename T, bool check = false>
static __device__ inline bool d_ZEencode2wordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warps = TPB / WS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // count non-zero values and output bitmaps
  const int tid1 = tid * 2;
  const int tid2 = tid1 + 1;
  const bool active1 = (tid1 < insize);
  const bool active2 = (tid2 < insize);
  const T val1 = active1 ? in[tid1] : 0;
  const T val2 = active2 ? in[tid2] : 0;
  const bool havenonzeroval1 = (active1 && (val1 != 0));
  const bool havenonzeroval2 = (active2 && (val2 != 0));
#if defined(WS) && (WS == 64)
  const unsigned long long temp = __ballot_sync(~0, havenonzeroval1);
  const int bm1 = (lane < 32) ? (int)temp : (int)(temp >> 32);  
  const unsigned long long temp1 = __ballot_sync(~0, havenonzeroval2);
  const int bm2 = (lane < 32) ? (int)temp1 : (int)(temp1 >> 32);
#else    
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
#endif 
  const int cnt = __popc(bm1) + __popc(bm2);
  const int comb = havenonzeroval1 + havenonzeroval2 * 2;
#if defined(WS) && (WS == 64)
  const int sublane = lane & 31;
  const int tmp1 = __shfl_sync(~0, comb, sublane / 2, 32) >> (lane % 2);
  const unsigned long long temp2 = __ballot_sync(~0, tmp1 & 1);
  const int bmlo = (lane < 32) ? (int)temp2 : (int)(temp2 >> 32);
#else
  const int sublane = lane;
  const int tmp1 = __shfl_sync(~0, comb, lane / 2) >> (lane % 2);
  const int bmlo = __ballot_sync(~0, tmp1 & 1);
#endif
#if defined(WS) && (WS == 64)
  const int tmp2 = __shfl_sync(~0, comb, 16 + sublane / 2, 32) >> (lane % 2);
  const unsigned long long temp3 = __ballot_sync(~0, tmp2 & 1);
  const int bmhi = (lane < 32) ? (int)temp3 : (int)(temp3 >> 32);
#else
  const int tmp2 = __shfl_sync(~0, comb, 16 + lane / 2) >> (lane % 2);
  const int bmhi = __ballot_sync(~0, tmp2 & 1);
#endif
  const int subwarps = TPB / 32;
#if defined(WS) && (WS == 64)
  const int subwarp = threadIdx.x / 32;
  if ((((__ballot_sync(~0, active1) >> (lane & 32)) & 0xffff'ffff) != 0) && (sublane % 8 == 0)) bmout_b[subwarp * 8 + sublane / 8] = bmlo >> sublane;
  if ((((__ballot_sync(~0, active2) >> (lane & 32)) & 0xffff'ffff) != 0) && (sublane % 8 == 0)) bmout_b[subwarp * 8 + sublane / 8 + 4] = bmhi >> sublane;
#else
  const int subwarp = warp;
  if (__any_sync(~0, active1) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8] = bmlo >> lane;
  if (__any_sync(~0, active2) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8 + 4] = bmhi >> lane;
#endif 
  if constexpr (sizeof(T) > 1) {  //MB: (never used) maybe somewhere else zero out last word of bitmap before a barrier and calling this function?
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  // output non-zero values
  int loc = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1));
  if (havenonzeroval1) dataout[loc++] = val1;
  if (havenonzeroval2) dataout[loc] = val2;

  datasize = temp_w[subwarps - 1];
  return true;
}


//warp-based four words per thread
template <typename T, bool check = false>
static __device__ inline bool d_ZEencode4wordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // count non-zero values and output bitmaps
  const int tid1 = tid * 4;
  const int tid2 = tid1 + 1;
  const int tid3 = tid2 + 1;
  const int tid4 = tid3 + 1;
  const bool active1 = (tid1 < insize);
  const bool active2 = (tid2 < insize);
  const bool active3 = (tid3 < insize);
  const bool active4 = (tid4 < insize);
  const T val1 = active1 ? in[tid1] : 0;
  const T val2 = active2 ? in[tid2] : 0;
  const T val3 = active3 ? in[tid3] : 0;
  const T val4 = active4 ? in[tid4] : 0;
  const bool havenonzeroval1 = (active1 && (val1 != 0));
  const bool havenonzeroval2 = (active2 && (val2 != 0));
  const bool havenonzeroval3 = (active3 && (val3 != 0));
  const bool havenonzeroval4 = (active4 && (val4 != 0));
#if defined(WS) && (WS == 64)
  const unsigned long long temp1 = __ballot_sync(~0, havenonzeroval1);
  const int bm1 = (lane < 32) ? (int)temp1 : (int)(temp1 >> 32);
  const unsigned long long temp2 = __ballot_sync(~0, havenonzeroval2);
  const int bm2 = (lane < 32) ? (int)temp2 : (int)(temp2 >> 32);
  const unsigned long long temp3 = __ballot_sync(~0, havenonzeroval3);
  const int bm3 = (lane < 32) ? (int)temp3 : (int)(temp3 >> 32);
  const unsigned long long temp4 = __ballot_sync(~0, havenonzeroval4);
  const int bm4 = (lane < 32) ? (int)temp4 : (int)(temp4 >> 32);
#else  
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
  const int bm3 = __ballot_sync(~0, havenonzeroval3);
  const int bm4 = __ballot_sync(~0, havenonzeroval4);
#endif  
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  const int comb = havenonzeroval1 + havenonzeroval2 * 2 + havenonzeroval3 * 4 + havenonzeroval4 * 8;
#if defined(WS) && (WS == 64)
  const int sublane = lane & 31;
  const int tmp1 = __shfl_sync(~0, comb, sublane / 4, 32) >> (lane % 4);
  const unsigned long long temp5 = __ballot_sync(~0, tmp1 & 1);
  const int bmA = (lane < 32) ? (int)temp5 : (int)(temp5 >> 32);
#else
  const int sublane = lane;    
  const int tmp1 = __shfl_sync(~0, comb, lane / 4) >> (lane % 4);
  const int bmA = __ballot_sync(~0, tmp1 & 1);
#endif  
#if defined(WS) && (WS == 64)
  const int tmp2 = __shfl_sync(~0, comb, 8 + sublane / 4, 32) >> (lane % 4);
  const unsigned long long temp6 = __ballot_sync(~0, tmp2 & 1);
  const int bmB = (lane < 32) ? (int)temp6 : (int)(temp6 >> 32);
#else  
  const int tmp2 = __shfl_sync(~0, comb, 8 + lane / 4) >> (lane % 4);
  const int bmB = __ballot_sync(~0, tmp2 & 1);
#endif 
#if defined(WS) && (WS == 64)
  const int tmp3 = __shfl_sync(~0, comb, 16 + sublane / 4, 32) >> (lane % 4);
  const unsigned long long temp7 = __ballot_sync(~0, tmp3 & 1);
  const int bmC = (lane < 32) ? (int)temp7 : (int)(temp7 >> 32);
#else
  const int tmp3 = __shfl_sync(~0, comb, 16 + lane / 4) >> (lane % 4);
  const int bmC = __ballot_sync(~0, tmp3 & 1);
#endif
#if defined(WS) && (WS == 64)
  const int tmp4 = __shfl_sync(~0, comb, 24 + sublane / 4, 32) >> (lane % 4);
  const unsigned long long temp8 = __ballot_sync(~0, tmp4 & 1);
  const int bmD = (lane < 32) ? (int)temp8 : (int)(temp8 >> 32);
#else
  const int tmp4 = __shfl_sync(~0, comb, 24 + lane / 4) >> (lane % 4);
  const int bmD = __ballot_sync(~0, tmp4 & 1);
#endif
const int subwarps = TPB / 32;
#if defined(WS) && (WS == 64)
  const int subwarp = threadIdx.x / 32;
  if ((((__ballot_sync(~0, active1) >> (lane & 32)) & 0xffff'ffff) != 0) && (sublane % 8 == 0)) bmout_b[subwarp * 16 + sublane / 8] = bmA >> sublane;
  if ((((__ballot_sync(~0, active2) >> (lane & 32)) & 0xffff'ffff) != 0) && (sublane % 8 == 0)) bmout_b[subwarp * 16 + sublane / 8 + 4] = bmB >> sublane;
  if ((((__ballot_sync(~0, active3) >> (lane & 32)) & 0xffff'ffff) != 0) && (sublane % 8 == 0)) bmout_b[subwarp * 16 + sublane / 8 + 8] = bmC >> sublane;
  if ((((__ballot_sync(~0, active4) >> (lane & 32)) & 0xffff'ffff) != 0) && (sublane % 8 == 0)) bmout_b[subwarp * 16 + sublane / 8 + 12] = bmD >> sublane;
#else
  const int subwarp = warp;
  if (__any_sync(~0, active1) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8] = bmA >> lane;
  if (__any_sync(~0, active2) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 4] = bmB >> lane;
  if (__any_sync(~0, active3) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 8] = bmC >> lane;
  if (__any_sync(~0, active4) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 12] = bmD >> lane;
#endif  
  if constexpr (sizeof(T) > 1) {  //MB: maybe somewhere else zero out last word of bitmap before a barrier and calling this function?
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  // output non-zero values
  int loc = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1)) + __popc(bm3 & ((1 << sublane) - 1)) + __popc(bm4 & ((1 << sublane) - 1));
  if (havenonzeroval1) dataout[loc++] = val1;
  if (havenonzeroval2) dataout[loc++] = val2;
  if (havenonzeroval3) dataout[loc++] = val3;
  if (havenonzeroval4) dataout[loc] = val4;

  datasize = temp_w[subwarps - 1];
  return true;
}


//thread-based X words per thread, X must be 8, 16, or 32
template <int X, typename T, bool check = false>
static __device__ inline bool d_ZEencodeXwordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  assert((X == 8) || (X == 16) || (X == 32));

  // count non-zero values and output bitmaps
  const int WPT = X;  // words per thread
  const int tid = threadIdx.x;
  int bmp = 0, cnt = 0;
  if (tid * WPT < insize) {
    for (int i = 0; i < WPT; i++) {
      const T val = in[tid * WPT + i];
      bmp |= (val != 0) << i;
    }
    if (tid * WPT - (insize - WPT) > 0) {
      bmp &= ~(-1 << (insize % WPT));
    }
    if constexpr (X == 8) ((byte*)bmout)[tid] = bmp;
    if constexpr (X == 16) ((short*)bmout)[tid] = bmp;
    if constexpr (X == 32) ((int*)bmout)[tid] = bmp;
    cnt = __popc(bmp);
  }

  // pad with zeros if necessary to alignment point
  if constexpr (sizeof(T) * 8 > X) {  //MB: maybe somewhere else zero out last word of bitmap before a barrier and calling this function?
    if (tid < WS) {
      const int base = (insize + (X - 1)) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) ((byte*)bmout)[base + tid] = 0;
    }
  }

  // compute prefix sum
  int pos = block_prefix_sum(cnt, temp_w);
  if (tid == TPB - 1) temp_w[WS] = pos;
  if constexpr (check) {
    if (__syncthreads_or(pos > datasize)) return false;
  } else {
    __syncthreads();
  }
  pos -= cnt;

  // output non-zero values
  if (bmp != 0) {
    for (int i = 0; i < WPT; i++) {
      if ((bmp >> i) & 1) {
        const T val = in[tid * WPT + i];
        dataout[pos++] = val;
      }
    }
  }

  datasize = temp_w[WS];
  return true;
}


template <typename T, int maxsize = CS, bool check = false>  // maxsize in bytes
static __device__ inline bool d_ZEencode(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  assert((TPB & (TPB - 1)) == 0);
  assert((maxsize & (maxsize - 1)) == 0);
  assert(maxsize % sizeof(T) == 0);
  assert((maxsize / sizeof(T) % TPB == 0) || (maxsize / sizeof(T) < TPB));
  const int wordsperthread = maxsize / sizeof(T) / TPB;
  if constexpr ((sizeof(T) <= 2) && (maxsize > 2048)) {
    // special case for byte and short data
    return d_ZEencodebyteshort<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread <= 1) {
    // warp-based 1 word per thread
    return d_ZEencode1wordperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 2) {
    // warp-based 2 words per thread
    return d_ZEencode2wordsperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 4) {
    // warp-based 4 words per thread
    return d_ZEencode4wordsperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 8) {
    // thread-based 8 words per thread
    return d_ZEencodeXwordsperthread<8, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 16) {
    // thread-based 16 words per thread
    return d_ZEencodeXwordsperthread<16, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 32) {
    // thread-based 32 words per thread
    return d_ZEencodeXwordsperthread<32, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else {
    // unsupported
    __trap();
    return false;
  }
}


template <typename T, typename U>  // U must be int or smaller; if smaller, it must be unsigned
static __device__ inline void d_ZEdecode_specialized(const int decsize, const T* const datain, const U* const bmin_t, T* const out, int* const temp_w)  // all sizes in number of words
{
  const int subWS = 32;
  const int tid = threadIdx.x;
  const int subwarp = tid / subWS;
  const int subwarps = TPB / subWS;
  const int sublane = tid % subWS;
  int num = (decsize + subWS - 1) / subWS;  // number of subchunks (rounded up)
  if constexpr (sizeof(T) == 8) num += num & 1;  // next higher even value

  // count non-zero values
  const int beg = subwarp * num / subwarps;
  const int end = (subwarp + 1) * num / subwarps;
  int cnt = 0;

  for (int i = beg * (4 / sizeof(U)) + sublane; i < end * (4 / sizeof(U)); i += subWS) {
    const int bm = bmin_t[i];
    cnt += __popc(bm);
  }

  for (int i = 1; i < subWS; i *= 2) {
    cnt += __shfl_xor_sync(~0, cnt, i, subWS);
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  if (tid < WS) {
    const int lane = tid % WS;
    int sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output non-zero values based on bitmap
  int pos = temp_w[subwarp] - cnt;
  for (int i = beg; i < end; i++) {
    int bm;
    if constexpr (sizeof(U) == 1) {
      bm = (int)bmin_t[i * 4 + sublane / 8] << (sublane & ~7);
      bm |= __shfl_xor_sync(~0, bm, 8, subWS);
      bm |= __shfl_xor_sync(~0, bm, 16, subWS);
    }
    if constexpr (sizeof(U) == 2) {
      bm = (int)bmin_t[i * 2 + sublane / 16] << (sublane & ~15);
      bm |= __shfl_xor_sync(~0, bm, 16, subWS);
    }
    if constexpr (sizeof(U) == 4) {
      bm = bmin_t[i];
    }
    const int offs = __popc(bm & ((1 << sublane) - 1)) - (((bm >> sublane) & 1) ^ 1);
    const int loc = i * subWS + sublane;
    if (loc < decsize) out[loc] = ((bm >> sublane) & 1) ? datain[pos + offs] : (T)0;
    pos += __popc(bm);
  }
}


//warp-based one word per thread
template <typename T>
static __device__ inline void d_ZEdecode1wordperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // read bitmap and count non-zero values
  const bool active = (tid < decsize);
  const bool havenonzeroval = (active && ((bmin_b[tid / 8] >> (tid % 8)) & 1));
#if defined(WS) && (WS == 64)
  const unsigned long long tmp = __ballot_sync(~0, havenonzeroval);
  const int bm = (lane < 32) ? (int)tmp : (int)(tmp >> 32);
#else  
  const int bm = __ballot_sync(~0, havenonzeroval);
#endif
  const int cnt = __popc(bm);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output values
  if (active) {
    const int loc = temp_w[subwarp] - cnt + __popc(bm & ((1 << sublane) - 1)) - (havenonzeroval ^ 1);
    out[tid] = havenonzeroval ? datain[loc] : (T)0;
  }
}


//warp-based two words per thread
template <typename T>
static __device__ inline void d_ZEdecode2wordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // read bitmap and count non-zero values
  const int tid1 = tid * 2;
  const int tid2 = tid1 + 1;
  const bool active1 = (tid1 < decsize);
  const bool active2 = (tid2 < decsize);
  const byte b = active1 ? (bmin_b[tid1 / 8] >> (tid1 % 8)) : 0;
  const bool havenonzeroval1 = (active1 && (b & 1));
  const bool havenonzeroval2 = (active2 && (b & 2));
#if defined(WS) && (WS == 64)
  const unsigned long long temp = __ballot_sync(~0, havenonzeroval1);
  const int bm1 = (lane < 32) ? (int)temp : (int)(temp >> 32);
  const unsigned long long temp1 = __ballot_sync(~0, havenonzeroval2);
  const int bm2 = (lane < 32) ? (int)temp1 : (int)(temp1 >> 32);
#else
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
#endif
  const int cnt = __popc(bm1) + __popc(bm2);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output values
  const int common = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1));
  const int loc1 = common - (havenonzeroval1 ^ 1);
  const int loc2 = common + havenonzeroval1 - (havenonzeroval2 ^ 1);
  if (active1) out[tid1] = havenonzeroval1 ? datain[loc1] : (T)0;
  if (active2) out[tid2] = havenonzeroval2 ? datain[loc2] : (T)0;
}


//warp-based four words per thread
template <typename T>
static __device__ inline void d_ZEdecode4wordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // read bitmap and count non-zero values
  const int tid1 = tid * 4;
  const int tid2 = tid1 + 1;
  const int tid3 = tid2 + 1;
  const int tid4 = tid3 + 1;
  const bool active1 = (tid1 < decsize);
  const bool active2 = (tid2 < decsize);
  const bool active3 = (tid3 < decsize);
  const bool active4 = (tid4 < decsize);
  const byte b = active1 ? (bmin_b[tid1 / 8] >> (tid1 % 8)) : 0;
  const bool havenonzeroval1 = (active1 && (b & 1));
  const bool havenonzeroval2 = (active2 && (b & 2));
  const bool havenonzeroval3 = (active3 && (b & 4));
  const bool havenonzeroval4 = (active4 && (b & 8));
#if defined(WS) && (WS == 64)
  const unsigned long long temp = __ballot_sync(~0, havenonzeroval1);
  const int bm1 = (lane < 32) ? (int)temp : (int)(temp >> 32);  
  const unsigned long long temp1 = __ballot_sync(~0, havenonzeroval2);
  const int bm2 = (lane < 32) ? (int)temp1 : (int)(temp1 >> 32);
  const unsigned long long temp2 = __ballot_sync(~0, havenonzeroval3);
  const int bm3 = (lane < 32) ? (int)temp2 : (int)(temp2 >> 32);
  const unsigned long long temp3 = __ballot_sync(~0, havenonzeroval4);
  const int bm4 = (lane < 32) ? (int)temp3 : (int)(temp3 >> 32);
#else  
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
  const int bm3 = __ballot_sync(~0, havenonzeroval3);
  const int bm4 = __ballot_sync(~0, havenonzeroval4);
#endif
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output values
  const int common = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1)) + __popc(bm3 & ((1 << sublane) - 1)) + __popc(bm4 & ((1 << sublane) - 1));
  const int loc1 = common - (havenonzeroval1 ^ 1);
  const int loc2 = common + havenonzeroval1 - (havenonzeroval2 ^ 1);
  const int loc3 = common + havenonzeroval1 + havenonzeroval2 - (havenonzeroval3 ^ 1);
  const int loc4 = common + havenonzeroval1 + havenonzeroval2 + havenonzeroval3 - (havenonzeroval4 ^ 1);
  if (active1) out[tid1] = havenonzeroval1 ? datain[loc1] : (T)0;
  if (active2) out[tid2] = havenonzeroval2 ? datain[loc2] : (T)0;
  if (active3) out[tid3] = havenonzeroval3 ? datain[loc3] : (T)0;
  if (active4) out[tid4] = havenonzeroval4 ? datain[loc4] : (T)0;
}


//thread-based X words per thread, X must be 8, 16, or 32
template <int X, typename T>
static __device__ inline void d_ZEdecodeXwordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  assert((X == 8) || (X == 16) || (X == 32));

  // read bitmap and count non-zero values
  const int WPT = X;  // words per thread
  const int tid = threadIdx.x;
  int bmp, cnt = 0;
  if (tid * WPT < decsize) {
    if constexpr (X == 8) bmp = ((byte*)bmin)[tid];
    if constexpr (X == 16) bmp = ((unsigned short*)bmin)[tid];
    if constexpr (X == 32) bmp = ((int*)bmin)[tid];
    cnt = __popc(bmp);
  }

  // compute prefix sum
  int pos = block_prefix_sum(cnt, temp_w) - cnt;

  // output values
  if (tid * WPT < decsize) {
    if ((tid | 31) * WPT + (WPT - 1) < decsize) {
      for (int i = 0; i < WPT; i++) {
        T val = 0;
        if ((bmp >> i) & 1) val = datain[pos++];
        out[tid * WPT + i] = val;
      }
    } else {
      for (int i = 0; i < WPT; i++) {
        if (tid * WPT + i >= decsize) break;
        T val = 0;
        if ((bmp >> i) & 1) val = datain[pos++];
        out[tid * WPT + i] = val;
      }
    }
  }
}


template <typename T, int maxsize = CS>  // maxsize in bytes
static __device__ inline void d_ZEdecode_small(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  assert((TPB & (TPB - 1)) == 0);
  assert((maxsize & (maxsize - 1)) == 0);
  assert(maxsize % sizeof(T) == 0);
  assert((maxsize / sizeof(T) % TPB == 0) || (maxsize / sizeof(T) < TPB));
  const int wordsperthread = maxsize / sizeof(T) / TPB;
  if constexpr (wordsperthread <= 1) {
    // warp-based 1 word per thread
    d_ZEdecode1wordperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 2) {
    // warp-based 2 words per thread
    d_ZEdecode2wordsperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 4) {
    // warp-based 4 words per thread
    d_ZEdecode4wordsperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 8) {
    // thread-based 8 words per thread
    d_ZEdecodeXwordsperthread<8, T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 16) {
    // thread-based 16 words per thread
    d_ZEdecodeXwordsperthread<16, T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 32) {
    // thread-based 32 words per thread
    d_ZEdecodeXwordsperthread<32, T>(decsize, datain, bmin, out, temp_w);
  } else {
    // unsupported
    __trap();
  }
}


template <typename T, int maxsize = CS>
static __device__ inline void d_ZEdecode(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  if constexpr (maxsize <= 2048) {
//    assert(false);
    d_ZEdecode_small<T, maxsize>(decsize, datain, bmin, out, temp_w);
  } else if ((sizeof(T) >= 4) /*|| (((size_t)bmin & 3) == 0)*/) {  // at least int aligned
//    assert(false);
    d_ZEdecode_specialized(decsize, datain, (int*)bmin, out, temp_w);
  } else if constexpr (sizeof(T) == 2) {  // short aligned
//    assert(false);
    const int tid = threadIdx.x;
    const int num = (decsize + 15) / 16;  // number of subchunks (rounded up)

    // count non-zero values
    const int beg = tid * num / TPB;
    const int end = (tid + 1) * num / TPB;
    int cnt = 0;
    for (int i = beg; i < end; i++) {
      const unsigned short bm = bmin[i];
      cnt += __popc((int)bm);
    }
    int pos = block_prefix_sum(cnt, temp_w) - cnt;

    // output non-zero values based on bitmap
    for (int i = beg; i < end; i++) {
      const unsigned short bm = bmin[i];
      for (int j = 0; j < 16; j++) {
        short val = 0;
        if ((bm >> j) & 1) val = datain[pos++];
        if (i * 16 + j < decsize) out[i * 16 + j] = val;
      }
    }
  } else {  // byte aligned
    const int tid = threadIdx.x;
    const int num = (decsize + 7) / 8;  // number of subchunks (rounded up)
    long long* const out_l = (long long*)out;
    assert(num <= TPB * 4);

    // count non-zeros
    const int beg = tid * num / TPB;
    const int end = (tid + 1) * num / TPB;
    int bmp = 0;
    for (int i = beg; i < end; i++) {
      bmp |= (int)bmin[i] << (8 * (i - beg));
    }
    const int cnt = __popc(bmp);
    int pos = block_prefix_sum(cnt, temp_w) - cnt;

    // output non-zero values based on bitmap
    for (int i = beg; i < end; i++) {
      const byte bm = bmp >> (8 * (i - beg));
      long long lval = 0;
      for (int j = 0; j < 8; j++) {
        long long val = 0;
        if ((bm >> j) & 1) val = datain[pos++];
        lval |= val << (j * 8);
      }
      out_l[i] = lval;
    }
  }
}


#endif
