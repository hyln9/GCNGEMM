#!/usr/bin/env python
# -*- coding: utf-8 -*-

# gemm.py
#
# Copyright (c) 2016 Yule Hou
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division

import numpy as np
import argparse as ap
import pyopencl as cl


def magic(d):
    d = np.int32(d)
    ad = np.uint32(np.abs(d))
    two31 = np.uint32(0x80000000)
    t = two31 + np.right_shift(np.uint32(d), np.uint32(31))
    anc = t - np.uint32(1) - t % ad
    p = np.int32(31)
    q1 = two31 // anc
    r1 = two31 - q1 * anc
    q2 = two31 // ad
    r2 = two31 - q2 * ad
    while True:
        p = p + np.int32(1)
        q1 = np.uint32(2) * q1
        r1 = np.uint32(2) * r1
        if r1 >= anc:
            q1 = q1 + np.uint32(1)
            r1 = r1 - anc
        q2 = np.uint32(2) * q2
        r2 = np.uint32(2) * r2
        if r2 >= ad:
            q2 = q2 + np.uint32(1)
            r2 = r2 - ad
        delta = ad - r2
        if not ((q1 < delta) or (q1 == delta and r1 == np.uint32(0))):
            break
    m = np.int32(q2 + np.uint32(1))
    if d < np.int32(0):
        m = -m
    s = p - np.int32(32)
    return m, s


class Hgemm:
    """
    Class which holds tuning parameters and acts as a kernel launcher
    """
    def __init__(self, filename, ctx, dev):
        with open(filename, 'rb') as f:
            prog_binary = f.read()
        self.prog  = cl.Program(ctx, dev, [prog_binary]).build()

    def __call__(self, queue, clA, clB, clC, wait=None):
        launch = getattr(self.prog, self.kernelName)
        return launch(queue, [self.gnumx*8, self.gnumy*8], [8, 8],
                      clA, clB, clC, self.ldA, self.ldB, self.ldC,
                      self.M, self.N, self.K, self.a, self.b, self.magic,
                      self.shift, wait_for=wait)

    def tune(self, M, N, K, ldA, ldB, ldC, alpha=1.0, beta=0.0, transA='N', transB='N'):
        self.M = np.uint32(M)
        self.N = np.uint32(N)
        self.K = np.uint32(K)
        self.ldA = np.uint32(ldA)
        self.ldB = np.uint32(ldB)
        self.ldC = np.uint32(ldC)
        self.a = np.float16(alpha)
        self.b = np.float16(beta)
        self.kernelName = ("hgemm_col_%(transA)s%(transB)s_64x64_8x8" % locals()).lower()
        self.gnumx = (self.M - 1) // 64 + 1
        self.gnumy = (self.N - 1) // 64 + 1
        self.magic, self.shift = magic(self.gnumy)


if __name__ == '__main__':

    parser = ap.ArgumentParser(usage='%(prog)s [options] <filename>',
        description='OpenDNN Fiji GEMM driver & benchmark.',
        formatter_class=lambda prog: ap.HelpFormatter(prog,max_help_position=27))
    parser.add_argument('filename', help='path to kernel binary file')
    parser.add_argument('-r', '--repeat', default=1, metavar='N',
        help='number of repeats for kernel execution', type=int)
    parser.add_argument('-tA', '--transA', default='N', metavar='TRANS',
        help='transpose op(A)', choices=['N'])
    parser.add_argument('-tB', '--transB', default='T', metavar='TRANS',
        help='transpose op(B)', choices=['T'])
    parser.add_argument('-m', default=6144, metavar='SIZE',
        help='matrix dimension m', type=int)
    parser.add_argument('-n', default=6144, metavar='SIZE',
        help='matrix dimension n', type=int)
    parser.add_argument('-k', default=6144, metavar='SIZE',
        help='matrix dimension k', type=int)
    parser.add_argument('-a', default=1.0, metavar='ALPHA',
        help='alpha', type=float)
    parser.add_argument('-b', default=0.0, metavar='BETA',
        help='beta', type=float)
    args = parser.parse_args()

    platforms  = filter(lambda p: 'AMD' in p.name, cl.get_platforms())
    devices    = filter(lambda d: 'Fiji' == d.name, platforms[0].get_devices())
    assert len(devices) == 1
    ctx   = cl.Context(devices)
    queue = cl.CommandQueue(ctx, properties=
        cl.command_queue_properties.PROFILING_ENABLE)

    hgemm = Hgemm(args.filename, ctx, devices)
    hgemm.tune(args.m, args.n, args.k, args.m, args.n, args.m)

    #A = np.asfortranarray(np.random.rand(args.m, args.k).astype(np.float16))
    #B = np.asfortranarray(np.random.rand(args.n, args.k).astype(np.float16))
    #A = np.asfortranarray(np.tril(np.full((args.m, args.k),1.0)).astype(np.float16))
    #B = np.asfortranarray(np.triu(np.full((args.n, args.k),1.0)).astype(np.float16))
    A = np.asfortranarray(np.random.randint(0, high=3, size=(args.m, args.k)).astype(np.float16))
    B = np.asfortranarray(np.random.randint(0, high=3, size=(args.n, args.k)).astype(np.float16))
    #A = np.ones((args.m, args.k), dtype=np.float16, order='F')
    #B = np.ones((args.n, args.k), dtype=np.float16, order='F')
    C = np.zeros((args.m, args.n), dtype=np.float16, order='F')
    V = np.copy(C)
    mf = cl.mem_flags
    clA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    clB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    clC = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=C)

    queue.finish()
    for i in xrange(args.repeat):
        event = hgemm(queue, clA, clB, clC)
        #V = args.a * np.dot(A, B)
        event.wait()
        print((event.profile.end - event.profile.start) / 1000000.0)
    cl.enqueue_copy(queue, C, clC).wait()
    #np.savetxt("diff.txt", C - V, fmt='%.10e')
