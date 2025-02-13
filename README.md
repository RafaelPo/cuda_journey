# Documenting progress for learning CUDA

Documenting my progress with CUDA. I code in Google Colab and [LeetGPU](https://leetgpu.com/) and get a lot of help from ChatGPT.

## Resources
- [gpu-mode lectures](https://github.com/gpu-mode/lectures/tree/main?tab=readme-ov-file)
  - Note: Started with Lecture 3
- Simon Boehm's blog post on optimising CUDA matmul kernels [here](https://siboehm.com/articles/22/CUDA-MMM)
- PMPP Book: [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.co.uk/Programming-Massively-Parallel-Processors-Hands/dp/0123814723)
  - There's a youtube channel associated with the book [here](https://www.youtube.com/@pmpp-book) - not sure if this is an official channel!

## Plan

As a start I want to do the following:

- Go through Simon Boehm's first 6 kernels:
  - [x] Naive 
  - [x] GMEM Coalescing
  - [x] SMEM Caching
  - [ ] 1D Blocktiling
  - [ ] 2D Blocktiling
  - [ ] Vectorized Mem Access
- Watch first few gpu-mode lectures
  1. [ ]
  2. [ ]
  3. [x]
  4. [ ]
  5. [x]

For later a few topics that I want to target:
- Code something multi-gpu 
- Understand (and possibly implement!) attention and flash-attention!
- Understand Triton

## Updates

### Lesson 1 -> SGEMM
See `code/simple_sgemm/`.

*Summary:* I first listened to Jeremy Howard's lecture while going through the same notebook where he explains two algorithms: 1) RGB-to-Greyscale conversion and 2) matrix multiplication. Following this I wanted something slightly different so I implemented SGEMM in google colab. Lastly I modified the code slightly so that I can run it on LeetGPU.

*Relevant resources:*
- Jeremy Howard's GPU-mode lecture ([link](https://www.youtube.com/watch?v=4sgKnKbR-WE)) with the notebook [here](https://github.com/gpu-mode/lectures/blob/main/lecture_003/pmpp.ipynb)
- `Kernel 1: Naive Implementation` section from Simon Boehm's blog post on optimising CUDA matmul kernels [here](https://siboehm.com/articles/22/CUDA-MMM)

### Lesson 2 -> SGEMM with shared memory
See `code/sgemm_with_shared_memory`

**Summary:** I listened to Jeremy Howard's second lecture where he talks about shared memory. I also found a new really good blog [post](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) by Lei Mao who goes through SGEMM optimisations. The notebook in the directory contains three kernels:
- `sgemm_kernel` -> From lesson 1
- `sgemm_kernel_reverse` -> Almost identical to `sgemm_kernel` but with a slight modification to indexing.
- `sgemm_with_shared_memory` -> Builds on `sgemm_kernel`. First copies data from matrices A and B to shared memory and then computes the inner product in tiles.

Over 1000 runs with dim=1024 the 3 approaches perform as follows:
- `sgemm_kernel` -> 2.61 ms ± 40.1
- `sgemm_kernel_reverse` -> 5.06 ms ± 77.9 µs
- `sgemm_kernel_with_shared_memory` -> 18.5 ms ± 19.6 µs

**Why does `sgemm_kernel` perform better than `sgemm_kernel_reverse`?**

From the CUDA C Best Practices guide ([link](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory:~:text=memory%20is%20modified.-,9.2.1.%20Coalesced%20Access%20to%20Global%20Memory,%EF%83%81,-A%20very%20important))

>A very important performance consideration in programming for CUDA-capable GPU architectures is the coalescing of global memory accesses. Global memory loads and stores by threads of a warp are coalesced by the device into as few as possible transactions.

Within each block, threads are grouped into warps, where each warp consists of 32 threads. Threads within a warp execute the same command. When threads within the same warp make memory requests an attempt to coalesce them is made that could lead to a reduced number of requests in the case of access requests pointing to neighbouring indices. Essentially, when a thread tries to access the element of an array at position `idx`, the response will not be just `array[idx]` but it would rather be a whole "cache line". For example, if the cache line is 128 bytes and our array is `float` that means 32 elements will be returned. Stated otherwise, when a thread accesses global memory, the memory request is typically fulfilled in chunks of 32 elements (for 4-byte floats) due to the memory coalescing behavior of the warp.

Looking at the kernels now, they only differ on the definition of `col` and `row`.

`__global__ void sgemm_kernel`
```cpp
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    ...
    matrix_c[row*dim + col] = alpha * tmp + beta * matrix_c[row*dim + col];
```

`__global__ void sgemm_kernel_reverse`
```cpp
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    ...
    matrix_c[row*dim + col] = alpha * tmp + beta * matrix_c[row*dim + col];
```


An important detail in how threads are id'd is that x is the "fast" index with y coming next and z being last, i.e., (Assuming `tpb(32,32)`.)
1. (x=0, y=0, z=0)
2. (x=1, y=0, z=0)
32. (x=31, y=0, z=0)
33. (x=0, y=1, z=0)


Looking at the two snippets of code above we see that in `sgemm_kernel` `matrix_c` memory requests are consecutive! 
1. (x=0, y=0, z=0)      -> `matrix_c[0]`
2. (x=1, y=0, z=0)      -> `matrix_c[1]`
3. (x=2, y=0, z=0)      -> `matrix_c[2]`
32. (x=31, y=0, z=0)     -> `matrix_c[31]`
33. (x=0, y=1, z=0)


While for `sgemm_kernel_reverse` we get:
1. (x=0, y=0, z=0)      -> `matrix_c[0]`
2. (x=1, y=0, z=0)      -> `matrix_c[dim]`
3. (x=2, y=0, z=0)      -> `matrix_c[2*dim]`
32. (x=31, y=0, z=0)     -> `matrix_c[31*dim]`
33. (x=0, y=1, z=0)

**Why does `sgemm_kernel_with_shared_memory` perform better than `sgemm_kernel`?**
Shared memory is much faster than global memory. 

For SGEMM we need a minimum of `3 * N^2` global reads (assuming A, B and C are square of dimension N). Now in the `sgemm_kernel` implementation each thread will perform `2N + 1` global reads and a single global write. We have `N^2` threads so that leaves us with `2 * N^3 + N^2` global reads and `N^2` global writes. For the `sgemm_kernel_with_shared_memory` each thread will first copy one element from each matrix from global memory to shared memory (3x global read per thread). Now instead of reading from global memory to perform the operations the threads read from shared memory. So in this case we need `3 * N^2` global reads.

**Relevant resources:**
- Jeremy Howard's GPU-mode lecture ([link](https://www.youtube.com/watch?v=eUuGdh3nBGo))
- The sections on kernels 2&3 from Simon Boehm's blog post on optimising CUDA matmul kernels [here](https://siboehm.com/articles/22/CUDA-MMM)
- Lei Mao's post on optimising SGEMM ([link](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/))