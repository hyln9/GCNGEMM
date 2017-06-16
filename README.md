# GCNGEMM
Optimized half precision gemm assembly kernels on AMD Fiji

## Introduction
This all date back to 2016 when I was trying to understand modern GPU architecture while studying performance optimization techniques for deep learning. Basically, I found out that technicians from NVIDIA have hand-assembled their cuBLAS and cuDNN routines and gained better utilization of their GPU products than AMD’s. Thankfully, [MaxAs](https://github.com/NervanaSystems/maxas) from Scott Gray revealed all those techniques to everyone and I began to consider possibilities of a corresponding GCN implementation.

After a quick investigation of [AMD GCN3 ISA architecture](http://gpuopen.com/compute-product/amd-gcn3-isa-architecture-manual/) and [CLRX](https://github.com/CLRX/CLRX-mirror), I came to the conclusion that it is possible to build optimized assembly kernels for GCN3, so I bought an AMD R9 Nano, an external GPU socket for my notebook and started my journey. After one month, I finished my first kernel and a modified overclock tool to allow 225w power consumption for R9 Nano in October 2016. Out of my expectation, my kernel can achieve around 8 teraflops on a 1.0 GHz 225w R9 Nano for a large square half precision matrix multiplication, that is, near 97.6% utilization (half precision runs at the same speed as single precision on GCN3). The result is encouraging since it’s much faster than clBLAS and we can beat a Titan X Maxwell with cuBLAS.

This repo is basically a snapshot of my work at that time. It contains a certain type of hgemm kernel and a test wrapper, both written for AMD OpenCL 2.0 fglrx driver (which can be ported to ROCm easily I think). I hope it’s useful for other people with the same focus on deep learning and GPGPU programming. If you are interested or have any ideas or questions, feel free to contact me via my email virgil#zju.edu.cn (replace # with @). I’ll write a detailed technical explanation of my code including some unique techniques if many people are interested.

GCNGEMM is not affiliated, sponsored, or otherwise endorsed by Advanced Micro Devices, Inc.

## Requirements
CLRX assembler v0.1.2: https://github.com/CLRX/CLRX-mirror/releases/tag/0.1.2

AMD OpenCL 2.0 fglrx driver: see https://community.amd.com/thread/202821 for details (this modified driver is made by other people)

PyOpenCL: https://github.com/pyopencl/pyopencl

Forked amdcovc for lifting power limit: https://github.com/hyln9/amdcovc

## Usage
```
./kernel.pl
./gemm.py [options] output.clo
```

