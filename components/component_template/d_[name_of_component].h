static __device__ inline bool d_[name_of_component](int& csize, byte in [CS], byte out [CS], byte temp [CS])  // GPU encoder
{
  // transforms the first csize bytes of the 'in' array and writes the result to the 'out' array
  // the transformation must be lossless
  // returns false if the encoded data does not fit in the out array
  // updates csize if the encoded data has a different size than the input data

  // must be thread-block-local code
  // is allowed to change the contents of all three arrays
  // must not allocate any __shared__ memory (use the temp array instead, e.g., long long* buf = (long long*)&temp;)
  // the three arrays are guaranteed to start at an 8-byte aligned address
}


static __device__ inline void d_i[name_of_component](int& csize, byte in [CS], byte out [CS], byte temp [CS])  // GPU decoder
{
  // transforms the first csize bytes of the 'in' array and writes the result to the 'out' array
  // the transformation must be lossless
  // updates csize if the decoded data has a different size than the input data

  // must be thread-block-local code
  // is allowed to change the contents of all three arrays
  // must not allocate any __shared__ memory (use the temp array instead, e.g., int* buf = (int*)&temp;)
  // the three arrays are guaranteed to start at an 8-byte aligned address
}
