static inline void h_[name_of_preprocessor](int& size, byte*& data, const int paramc, const double paramv [])  // CPU preprocessor encoder
{
  // transforms the 'size' bytes in the 'data' array and writes the result either back to the 'data' array or to a new array and then makes 'data' point to this new array
  // if the number of bytes changes, the 'size' needs to be updated accordingly
  // the data array must start at an 8-byte aligned address
  // 'paramc' specifies the number of elements in the 'paramv' array
  // the 'paramv' array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.)
  // this code must be manually parallelized (using OpenMP) if desired
}

static inline void h_i[name_of_preprocessor](int& size, byte*& data, const int paramc, const double paramv [])  // CPU preprocessor decoder
{
  // transforms the 'size' bytes in the 'data' array and writes the result either back to the 'data' array or to a new array and then makes 'data' point to this new array
  // if the number of bytes changes, the 'size' needs to be updated accordingly
  // the data array must start at an 8-byte aligned address
  // 'paramc' specifies the number of elements in the 'paramv' array
  // the 'paramv' array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.)
  // this code must be manually parallelized (using OpenMP) if desired
}
