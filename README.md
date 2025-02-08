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
  - [ ] GMEM Coalescing
  - [ ] SMEM Caching
  - [ ] 1D Blocktiling
  - [ ] 2D Blocktiling
  - [ ] Vectorized Mem Access
- Watch first few gpu-mode lectures
  1. [ ]
  2. [ ]
  3. [x]
  4. [ ]

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
