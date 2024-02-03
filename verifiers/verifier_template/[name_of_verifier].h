static void [name_of_verifier](const int size, const byte* const __restrict__ recon, const byte* const __restrict__ orig, const int paramc, const double paramv [])
{
  // checks whether the reconstructed (recon) data is within the given error bound of the original (orig) data
  // the verifier should print an error message and exit if it finds that the reconstructed data does not meet the specified error bound
  // otherwise, the verifier should print a message stating that the verification passed

  // the two arrays are guaranteed to start at an 8-byte aligned address
  // must be host-only code
  // the size [in bytes] pertains to both the recon and the orig arrays
  // paramc specifies the number of elements in the paramv array
  // the paramv array passes the command-line arguments provided to this verifier (e.g., the error bound)
} 

