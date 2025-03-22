static inline void d_[name_of_preprocessor](int& long long, byte*& data, const int paramc, const double paramv [])  // GPU preprocessor encoder
{
  // transforms the 'size' bytes in the 'data' array and writes the result either back to the 'data' array or to a new array and then makes 'data' point to this new array
  // if the number of bytes changes, the 'size' needs to be updated accordingly
  // the data array must start at an 8-byte aligned address
  // 'paramc' specifies the number of elements in the 'paramv' array
  // the 'paramv' array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.)
  // must be a host function that launches a kernel to do the preprocessing
  // the kernel is allowed to allocate and use shared memory
  // 'data' must be in device memory
}

static inline void d_i[name_of_preprocessor](long long& size, byte*& data, const int paramc, const double paramv [])  // GPU preprocessor decoder
{
  // transforms the 'size' bytes in the 'data' array and writes the result either back to the 'data' array or to a new array and then makes 'data' point to this new array
  // if the number of bytes changes, the 'size' needs to be updated accordingly
  // the data array must start at an 8-byte aligned address
  // 'paramc' specifies the number of elements in the 'paramv' array
  // the 'paramv' array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.)
  // must be a host function that launches a kernel to do the preprocessing
  // the kernel is allowed to allocate and use shared memory
  // 'data' must be in device memory
}
