# LC Framework

LC is a framework for automatically creating high-speed lossless and error-bounded lossy data compression and decompression algorithms.


## Overview

LC consists of the following three parts:
 - Component library
 - Preprocessor library
 - Framework

Both libraries contain encoders and decoders for CPU and GPU execution. The user can extend these libraries. The framework takes preprocessors and components from these libraries and chains them into a pipeline to build a compression algorithm. It similarly chains the corresponding decoders in the opposite order to build the matching decompression algorithm. Moreover, the framework can automatically search for effective algorithms for a given input file or set of files by testing sets of components in each pipeline stage.

---

## Quick-Start Guide and Tutorial

### Installation

To download LC, run the following Linux commands:

    git clone https://github.com/burtscher/LC-framework.git
    cd LC-framework/

If you want to run LC on the CPU, generate the framework as follows:

    ./generate_Host_LC-Framework.py

If, instead, you want to run LC on the GPU, generate the framework as follows:

    ./generate_Device_LC-Framework.py

In either case, run the printed command to compile the generated code. For the CPU, use:

    g++ -O3 -march=native -fopenmp -DUSE_CPU -I. -std=c++17 -o lc lc.cpp

For the GPU, use:

    nvcc -O3 -arch=sm_70 -DUSE_GPU -Xcompiler "-O3 -march=native -fopenmp" -I. -o lc lc.cu

You may have to adjust these commands and flags to your system and compiler. For instance, the *sm_70* should be changed to match your GPU's compute capability.

The generate_Hybrid_LC-Framework.py script is only for testing and should not be used as it generates slow code.


### Usage Examples for Lossless Compression Algorithms

The following examples assume you have a file called *input.dat* in the current directory and want to find a good compression algorithm for it. See below for a description of the available preprocessors and components.

Assume you believe that using bit shuffling (BIT) and run-length encoding (RLE) at 4-byte granularity make a good compressor. Then you can see how well it compresses by entering the following command (note the two pairs of quotes):

    ./lc input.dat CR "" "BIT_4 RLE_4"

This will produce output that lists the compression ratio at the end. If you want to see whether running the RLE component at 1-byte granularity performs better, try:

    ./lc input.dat CR "" "BIT_4 RLE_1"

To find out which components (and preprocessors) are available, simply run:

    ./lc

If you want to see more stats on the input and output data as well as throughput information in addition to the compression ratio, switch the mode from *CR* to *AL*:

    ./lc input.dat AL "" "BIT_4 RLE_1"

Note that using *AL* also turns on verification to make sure the decompressed data is bit-by-bit equivalent to the original data.

One of the key strengths of LC is its ability to automatically search for a good compression algorithm. For example, if you want LC to try all available components in the second stage, type:

    ./lc input.dat CR "" "BIT_4 .+"

The ".+" is a regular expression that matches the names of all components in the library. You can use it to select any subset of the available components. Of course, you can also use a regular expression for the first pipeline stage by entering:

    ./lc input.dat CR "" ".+ .+"

This is not limited to two stages. To search for the best 3-stage pipeline, use:

    ./lc input.dat CR "" ".+ .+ .+"

Note that the search time increases exponentially with the number of stages. Before you perform a search, you can check the size of the search space using the *PR* mode as follows:

    ./lc input.dat PR "" ".+ .+ .+ .+"

The output lists the number of algorithms that will be tested as well as which components will be considered in each pipeline stage. If this number is too large, i.e., the search would take too long, try reducing the search space by limiting the number of components to be considered:

    ./lc input.dat CR "" "DIFF_4 .+ .+ R.+|C.+|H.+"

If available, we recommend using the GPU version of LC as it tends to be much faster than the CPU version. To further speed up the search, LC includes a genetic algorithm (GA) to quickly search for a good but not necessarily the best algorithm. If you want to run the GA to find a good pipeline with 5 stages, enter the following command:

    ./scripts/ga_search.py -s 5 input.dat

If you are interested in the throughput in addition to the compression ratio, use the *EX* mode of LC like this:

    ./lc input.dat EX "" ".+ .+"

The output includes the Pareto front (https://en.wikipedia.org/wiki/Pareto_front) at the end, allowing the user to pick the best algorithm for a given compression or decompression throughput. The six columns list the algorithm, the compression ratio, the CPU compression throughput, the CPU decompression throughput, the GPU compression throughput, and the GPU decompression throughput. The throughputs are given in gigabytes per second.

All *CR* and *EX* runs with more than one algorithm also write their results to a CSV file that can be opened with most spreadsheet applications to view and postprocess the results.

*EX*- and *AL*-runs with one algorithm write the compressed data to a file called *LC.encoded* and the decompressed data to a file called *LC.decoded*.

In summary, LC supports the following modes:

 **AL**: This mode provides the most detailed output but only works for a single pipeline.

 **PR**: This mode prints the search space and then quits.

 **CR**: This mode searches for the best compressing algorithm.

 **EX**: This mode searches for the algorithms on the Pareto front, taking into account both the compression ratio and the compression/decompression throughput.

 **TS**: This mode is for testing only and should not be used.


### Usage Examples for Lossy Floating-Point Compression Algorithms

To generate lossy algorithms with LC, preprocessors are needed. They must be fully specified (no regular expressions are allowed) and cannot be searched for automatically as they require user-specified parameters such as the error bound.

To find a good lossy compression algorithm for IEEE-754 32-bit single-precision floating-point data that are quantized with a maximum point-wise absolute error bound of 0.01 and then losslessly compressed with three components, enter:

    ./lc input.dat CR "QUANT_ABS_0_f32(0.01)" ".+ .+ R.+|C.+|H.+"

To do the same with a point-wise relative error bound, use:

    ./lc input.dat CR "QUANT_REL_0_f32(0.01)" ".+ .+ R.+|C.+|H.+"

The preprocessors work with the *CR*, *EX*, and *AL* modes. However, since both *EX* and *AL* verify the result, the default lossless verification will likely fail for lossy compression. LC includes a set of verifiers that can be selected in lieu of the default verifier. For an point-wise absolute error bound of 0.001, use:

    ./lc input.dat EX "QUANT_ABS_0_f32(0.001)" ".+ R.+|C.+|H.+" "MAXABS_f32(0.001)"

See the ./verifiers/ directory for additional available verifiers or the description below.

These quantizers replace any lost bits with zeros. If you prefer those bits be replaced by random data to minimize autocorrelation, use:

    ./lc input.dat CR "QUANT_ABS_R_f32(0.01)" ".+ .+ R.+|C.+|H.+"

or

    ./lc input.dat CR "QUANT_REL_R_f32(0.01)" ".+ .+ R.+|C.+|H.+"



### Standalone Compressor and Decompressor Generation

Once you have determined a good lossless or lossy compression algorithm (e.g., "TUPL4_1 RRE_1 CLOG_1"), you can generate a standalone compressor and a standalone decompressor that are optimized for this algorithm.

To generate the CPU version, run:

    ./generate_standalone_CPU_compressor_decompressor.py "" "TUPL4_1 RRE_1 CLOG_1"

To generate the GPU version, run:

    ./generate_standalone_GPU_compressor_decompressor.py "" "TUPL4_1 RRE_1 CLOG_1"

In either case, run the printed commands to compile the generated code. For the CPU, use:

    g++ -O3 -march=native -fopenmp -I. -std=c++17 -o compress compressor-standalone.cpp
    g++ -O3 -march=native -fopenmp -I. -std=c++17 -o decompress decompressor-standalone.cpp

For the GPU, use:

    nvcc -O3 -arch=sm_70 -DUSE_GPU -Xcompiler "-march=native -fopenmp" -I. -o compress compressor-standalone.cu
    nvcc -O3 -arch=sm_70 -DUSE_GPU -Xcompiler "-march=native -fopenmp" -I. -o decompress decompressor-standalone.cu

You may have to adjust these commands and flags to your system and compiler. For instance, the *sm_70* should be changed to match your GPU's compute capability.

At this point, you can compress files with:

    ./compress input_file_name compressed_file_name [y]

and decompress them with:

    ./decompress compressed_file_name decompressed_file_name [y]

Both commands accept an optional "y" parameter at the end. If it is specified, the compressor and decompressor will measure and print the throughput.


---


## Available Components, Preprocessors, and Verifiers

All LC compression pipelines start with zero or more *preprocessors* and end with one or more *components*. The preprocessors require parentheses after their names and may take parameters. The components do not take any parameters and cannot have parentheses.


## Available Components

The LC framework breaks the input data up into chunks of 16 kB, each of which is compressed independently and in parallel using the selected components. All components are lossless. Most components support different word sizes. The number at the end of their names indicates the word size in bytes. For example, "_4" means the word size is 4 bytes (e.g., ints or floats). To structure the description of the components, we group them into the following four categories: mutators, shufflers, predictors, and reducers. The goal of the first three types is to better expose patterns so that the reducer components can compress the data more effectively. Only reducer components make sense the the last pipeline stage.


### Mutators

Mutators computationally transform each value. This is done independently of other values and does not compress the data.

**NUL**: This component performs the identity transformation, meaning it outputs the input verbatim. It is useful in that it allows longer pipelines to also cover algorithms with fewer stages.

**TCMS**: This component converts each value from twos-complement to magnitude-sign representation, which is often easier to compress because it tends to yield more leading zero bits.

**DBEFS**: This component operates on IEEE-754 floating-point values. It first de-biases the exponent and then rearranges the data fields from sign, exponent, fraction order to (de-biased) exponent, fraction, sign order.

**DBESF**: This component operates on IEEE-754 floating-point values. It first de-biases the exponent and then rearranges the data fields from sign, exponent, fraction order to (de-biased) exponent, sign, fraction order.


### Shufflers

Shufflers rearrange the order of the values but perform no computation on them. Some shufflers reorder the bits or bytes within a word. None of them compress the data.

**BIT**: This component is often referred to as "bit shuffle" or "bit transpose". It takes the most significant bit of each value in the input and outputs them together, then it takes the second most significant bit of each value and outputs them, and so on down to the least significant bit. This improves compressibility if the values tend to have the same bits in certain positions.

**TUPLk**: This component assumes the data to be a sequence of k-tuples, which it rearranges by listing all first tuple values, then all second tuple values, and so on. For example, a tuple size of k = 3 changes the linear sequence x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 into x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4. This is beneficial as values belonging to the same "dimension" often correlate more with each other than with other values from within the same tuple.


### Predictors

Predictors guess the next value by extrapolating it from prior values and then subtracting the prediction from the actual value, which yields a residual sequence. If the predictions are accurate, the residuals cluster around zero, making them easier to compress than the original data. Predictors per se do not compress the data.

**DIFF**: This component computes the difference sequence (also called "delta modulation") by subtracting the previous value from the current value and outputting the resulting difference. If neighboring values correlate with each other, this tends to produce a more compressible sequence.

**DIFFMS**: This component computes the difference sequence like DIFF does but outputs the result in sign-magnitude format, which is often more compressible because it tends to produce values with many leading zero bits.


### Reducers

Reducers are the only components that can compress the data. They exploit various types of redundancies to do so.

**CLOG**: This component breaks the data up into 32 subchunks, determines the smallest amount of leading zero bits of all values in a subchunk, records this count, and then stores only the remaining bits of each value. This compresses data with leading zero bits.

**HCLOG**: This component works like CLOG except it first applies the TCMS transformation to all values in a subchunk that yield no leading zero bits when using CLOG.

**RLE**: This component performs run-length encoding. It counts how many times a value appears in a row. Then it counts how many non-repeating values follow. Both counts are emitted and followed by a single instance of the repeating value as well as all non-repeating values.

**RRE**: This component creates a bitmap in which each bit specifies whether the corresponding word in the input is a repetition of the prior word or not. It outputs the non-repeating words and a compressed version of the bitmap that is repeatedly compressed with the same algorithm.

**RZE**: This component creates a bitmap in which each bit specifies whether the corresponding word in the input is zero or not. It outputs the non-zero words and a compressed version of the bitmap like RRE does.


## Available Preprocessors

Preprocessors operate on the entire data (i.e., there is no chunking) and can be lossy or lossless. Some preprocessors support different data types. The end of their names indicates the data type for which they are designed. For example, "_f32" means the preprocessor targets 32-bit floating-point values. To structure the description of the preprocessors, we group them into lossy or lossless preprocessors.

### Lossless

These quantizers support INFs and NaNs. The end of the quantizer name indicates the data type for which it is designed.

**NUL**: This preprocessor performs the identity transformation, meaning it outputs the input verbatim. It takes no parameters.

**LORxD**: This preprocessor performs an x-dimensional (x = 1, 2, or 3) Lorenzo transformation, i.e., it computes a multidimensional difference sequence. It takes x parameters specifying the size of the input along each dimension.


### Lossy Quantizers

All quantizers require a parameter that specifies the maximally allowed error bound EB. They take an optional second parameter specifying a threshold. Any value whose magnitude is at or above the threshold is compressed losslessly and not quantized. The quantizers support INFs and NaNs. The end of the quantizer name indicates the data type for which it is designed.

**QUANT_ABS_0**: These preprocessors quantize 32- and 64-bit floating-point values based on the provided point-wise absolute error bound. All values that end up in the same quantization bin are decompressed to the same value. These preprocessors guarantee that the original value V is decoded to a value V' such that V - EB <= V' <= V + EB.

**QUANT_ABS_R**: These preprocessors quantize 32- and 64-bit floating-point values based on the provided point-wise absolute error bound. Each value from the same quantization bin is decompressed to a random value within the provided error bound to minimize autocorrelation. These preprocessors guarantee that the original value V is decoded to a value V' such that V - EB <= V' <= V + EB.

**QUANT_R2R**: These preprocessors quantize 32- and 64-bit floating-point values just like their QUANT_ABS counterparts except the provided error bound is first multiplied by the range of values occurring in the input, where the range is the maximum value minus the minimum value.

**QUANT_REL_0**: These preprocessors quantize 32- and 64-bit floating-point values based on the provided point-wise relative error bound. All values that end up in the same quantization bin are decompressed to the same value. These preprocessors guarantee that the original value V is decoded to a value V' with the same sign such that |V| / (1 + EB) <= |V'| <= |V| \* (1 + EB).

**QUANT_REL_R**: These preprocessors quantize 32- and 64-bit floating-point values based on the provided point-wise relative error bound. Each value from the same quantization bin is decompressed to a random value within the provided error bound to minimize autocorrelation. These preprocessors guarantee that the original value V is decoded to a value V' with the same sign such that |V| / (1 + EB) <= |V'| <= |V| \* (1 + EB).


## Available Verifiers

Some verifiers support different data types. The end of their names indicates the data type for which they are designed.

**LOSSLESS**: This verifier is the default. It passes verification if the decompressed output matches every bit of the original input.

**PASS**: This verifier always passes verification and is only useful for debugging.

**MAXABS**: This verifier takes a point-wise absolute error bound as parameter and only passes verification if every output value is within the specified error bound.

**MAXR2R**: This verifier works like MAXABS except the provided error bound is first multiplied by the range of values occurring in the input, where the range is the maximum value minus the minimum value.

**MAXREL**: This verifier takes a point-wise relative error bound as parameter and only passes verification if every output value is within the specified error bound.

**MSE**: This verifier takes a mean squared error as parameter and only passes verification if the mean squared error of the output values is within the error bound.

**PSNR**: This verifier takes a peak-signal-to-noise ratio (PSNR) as parameter and only passes verification if the PSNR of the output values is above the specified lower bound.


---


## Adding and Removing Components

LC users can add and delete components. To remove a component from the library, simple delete the corresponding header files from the *components* subdirectory.


### Adding Your Own CPU Component

To add a CPU component, place a new header file in the *components* subdirectory whose file name must start with "h_" (for "host"). The header file must include the encoder and decoder functions and may include helper functions (with globally unique names). The name of the encoder function must be identical to the header file name without the extension. The name of the decoder function must be the same except it needs to include an "i" (for "inverse") after the first underscore. For example, the **NAME_4** component's header file for CPU execution must be named **h_NAME_4.h** and must contain an encoder function called **h_NAME_4** and a decoder function called **h_iNAME_4**.

The prototype of a CPU encoder is:

    static inline bool h_NAME_4(int& csize, byte in[CS], byte out[CS]);

This function returns false if the encoded data does not fit in the out array and true otherwise.

The prototype of a CPU decoder is:

    static inline void h_iNAME_4(int& csize, byte in[CS], byte out[CS]);

The encoder and decoder functions must losslessly transform the first csize bytes of the *in* array and write the result to the *out* array. They must update *csize* if the transformed data has a different size than the input. The code must be serial (e.g., it cannot use OpenMP) and cannot use global variables (or static local variables) so as not to interfere with LC's performance optimizations and automatic parallelization. Note that both functions are allowed to change the contents of the *in* and the *out* arrays. The two arrays are guaranteed to start at an 8-byte aligned address.

Templates for implementing a new CPU component are available in the *components/component_template* subdirectory.


### Adding Your Own GPU Component

To add a GPU component, place a new header file in the *components* subdirectory whose file name must start with "d_" (for "device"). The header file must include the encoder and decoder functions and may include helper functions (with globally unique names). The name of the encoder function must be identical to the header file name without the extension. The name of the decoder function must be the same except it needs to include an "i" (for "inverse") after the first underscore. For example, the **NAME_4** component's header file for GPU execution must be named **d_NAME_4.h** and must contain an encoder function called **d_NAME_4** and a decoder function called **d_iNAME_4**.

The prototype of a GPU encoder is:

    static __device__ inline bool d_NAME_4(int& csize, byte in[CS], byte out[CS], byte temp[CS]);

This function returns false if the encoded data does not fit in the out array and true otherwise.

The prototype of a GPU decoder is:

    static __device__ inline void d_iNAME_4(int& csize, byte in[CS], byte out[CS], byte temp[CS]);

The encoder and decoder functions must losslessly transform the first csize bytes of the *in* array and write the result to the *out* array. They must update *csize* if the transformed data has a different size than the input. The code must run in a single thread block and cannot use global variables so as not to interfere with LC's performance optimizations and automatic parallelization across thread blocks. Furthermore, the code must not allocate any "\_\_shared\_\_" memory. Instead, it should use the *temp* for obtaining shared memory (e.g., int\* buf = (int\*)&temp;). Note that both functions are allowed to change the contents of the *in*, the *out*, and the *temp* arrays. The three arrays are guaranteed to start at an 8-byte aligned address.

Templates for implementing a new GPU component are available in the *components/component_template* subdirectory.


### Example of a CPU Component

The following code provides a simple example of a CPU component called **INC_1** that adds 1 to each byte ("_1"). It can be invoked, for example, using the *./lc input EX "" "INC_1 .+"* command line.

    static inline bool h_INC_1(int& csize, const byte in [CS], byte out [CS])
    {
      for (int i = 0; i < csize; i++) {
        out[i] = in[i] + 1;
      }
      return true;
    }

    static inline void h_iINC_1(int& csize, const byte in [CS], byte out [CS])
    {
      for (int i = 0; i < csize; i++) {
        out[i] = in[i] - 1;
      }
    }


### Example of a GPU Component

The following code provides a simple example of a GPU component called **INC_1** that adds 1 to each byte ("_1"). /TPB/ stands for "threads per block" and is a predefined variable in LC. The component can be invoked, for example, using the *./lc input EX "" "INC_1 .+"* command line.

    static __device__ inline bool d_INC_1(int& csize, byte in [CS], byte out [CS], byte temp [CS])
    {
      for (int i = threadIdx.x; i < csize; i += TPB) {
        out[i] = in[i] + 1;
      }
      return true;
    }

    static __device__ inline void d_iINC_1(int& csize, byte in [CS], byte out [CS], byte temp [CS])
    {
      for (int i = threadIdx.x; i < csize; i += TPB) {
        out[i] = in[i] - 1;
      }
    }


---


## Adding and Removing Preprocessors

LC users can add and delete preprocessors. To remove a preprocessor from the library, simple delete the corresponding header files from the *preprocessors* subdirectory.


### Adding Your Own CPU Preprocessor

To add a CPU preprocessor, place a new header file in the *preprocessors* subdirectory whose file name must start with "h_" (for "host"). The header file must include the encoder and decoder functions and may include helper functions (with globally unique names). The name of the encoder function must be identical to the header file name without the extension. The name of the decoder function must be the same except it needs to include an "i" (for "inverse") after the first underscore. For example, the **PRE_f32** preprocessor's header file for CPU execution must be named **h_PRE_f32.h** and must contain an encoder function called **h_PRE_f32** and a decoder function called **h_iPRE_f32**.

The prototype of a CPU preprocessor encoder is:

    static inline void h_PRE_f32(int& size, byte\*& data, const int paramc, const double paramv[]);

The prototype of a CPU preprocessor decoder is:

    static inline void h_iPRE_f32(int& size, byte\*& data, const int paramc, const double paramv[]);

The encoder and decoder functions transform *size* bytes in the *data* array and write the result either back to the *data* array or to a new array and then make *data* point to this new array (and deallocate the old *data* array). If the number of bytes changes, the *size* must be updated accordingly. The *data* array must start at an 8-byte aligned address. The *paramc* argument specifies the number of elements in the *paramv* array. The *paramv* array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.). The two functions must be manually parallelized using OpenMP if desired.

Templates for implementing a new CPU preprocessor are available in the *preprocessors/preprocessor_template* subdirectory.


### Adding Your Own GPU Preprocessor

To add a GPU preprocessor, place a new header file in the *preprocessors* subdirectory whose file name must start with "d_" (for "device"). The header file must include the encoder and decoder functions and may include helper functions (with globally unique names). The name of the encoder function must be identical to the header file name without the extension. The name of the decoder function must be the same except it needs to include an "i" (for "inverse") after the first underscore. For example, the **PRE_f32** preprocessor's header file for GPU execution must be named **d_PRE_f32.h** and must contain an encoder function called **d_PRE_f32** and a decoder function called **d_iPRE_f32**.

The prototype of a GPU preprocessor encoder is:

    static inline void d_PRE_f32(int& size, byte\*& data, const int paramc, const double paramv[]);

The prototype of a GPU preprocessor decoder is:

    static inline void d_iPRE_f32(int& size, byte\*& data, const int paramc, const double paramv[]);

The encoder and decoder functions transform *size* bytes in the *data* array and write the result either back to the *data* array or to a new array and then make *data* point to this new array (and deallocate the old *data* array). If the number of bytes changes, the *size* must be updated accordingly. The *data* array must start at an 8-byte aligned address. The *paramc* argument specifies the number of elements in the *paramv* array. The *paramv* array passes the command-line arguments provided to this preprocessor (e.g., the error bound, data set dimensionality, etc.). The two functions run on the host and must invoke appropriate kernels to perform the preprocessing. The kernels are allowed to allocate and use shared memory. Note that the *data* array is allocated on the GPU and cannot be directly accessed from the host code.

Templates for implementing a new GPU preprocessor are available in the *preprocessors/preprocessor_template* subdirectory.


### Example of a CPU Preprocessor

The following code provides an example of a CPU preprocessor called **ADD_i32** that adds a user-provided constant to each 32-bit integer ("_i32"). It can be invoked, for example, using the *./lc input EX "ADD_i32(7)" ".+"* command line.

    static inline void h_ADD_i32(int& size, byte*& data, const int paramc, const double paramv [])
    {
      assert(paramc == 1);
      assert(size % sizeof(int) == 0);

      int* const idata = (int*)data;
      const int offset = paramv[0];

      #pragma omp parallel for default(none) shared(size, idata, offset)
      for (int i = 0; i < size / sizeof(int); i++) {
        idata[i] += offset;
      }
    }

    static inline void h_iADD_i32(int& size, byte*& data, const int paramc, const double paramv [])
    {
      assert(paramc == 1);
      assert(size % sizeof(int) == 0);

      int* const idata = (int*)data;
      const int offset = paramv[0];

      #pragma omp parallel for default(none) shared(size, idata, offset)
      for (int i = 0; i < size / sizeof(int); i++) {
        idata[i] -= offset;
      }
    }


---


## Notes

LC currently supports inputs of up to 2 GB in size.

LC currently only works on little-endian systems.

For testing, you can generate the LC framework using the *generate_Hybrid_LC-Framework.py* script. This will include all components and preprocessors for which both CPU and GPU versions exist, i.e., whose names match except for the leading "h_" and "d_". Running the resulting code will redundantly perform the compression and decompression on both devices and, importantly, compare the results bit for bit. This is useful to ensure that the CPU and GPU implementations of all components and preprocessors produce the exact same compressed and decompressed data.


---

## Team

The LC framework is being developed at Texas State University by Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, and Yiqian Liu under the supervision of Prof. Martin Burtscher and is joint work with Sheng Di and Franck Cappello from Argonne National Laboratory.


## Sponsor

This project is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.


## License

BSD 3-Clause
