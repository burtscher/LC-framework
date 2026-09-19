static inline bool h_[name_of_component](int& csize, byte in [CS], byte out [CS])  // CPU encoder
{
  // transforms the first csize bytes of the 'in' array and writes the result to the 'out' array
  // the transformation must be lossless
  // returns false if the encoded data does not fit in the out array
  // updates csize if the encoded data has a different size than the input data

  // must be serial code (e.g., cannot use OpenMP)
  // is allowed to change the contents of both arrays
  // the two arrays are guaranteed to start at an 8-byte aligned address
}


static inline void h_i[name_of_component](int& csize, byte in [CS], byte out [CS])  // CPU decoder
{
  // transforms the first csize bytes of the 'in' array and writes the result to the 'out' array
  // the transformation must be lossless
  // updates csize if the decoded data has a different size than the input data

  // must be serial code (e.g., cannot use OpenMP)
  // is allowed to change the contents of both arrays
  // the two arrays are guaranteed to start at an 8-byte aligned address
}

