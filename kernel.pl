#!/usr/bin/perl

# kernel.pl
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

use strict;
use IPC::Open3;
use kernels::hgemm_col_nt_64x64_8x8;

my $cmd = 'clrxasm -o output.clo';
my $pid = open3(*AS_SRC{IO}, *AS_OUT{IO}, *AS_ERR{IO}, $cmd);

print AS_SRC <<EOS;
.amd
.gpu Fiji
.64bit
.compile_options "Copyright (c) 2016 Yule Hou"
.driver_info "@(#) OpenCL 2.0 AMD-APP (1912.5).  Driver version: 1912.5 (VM)"
EOS

kernels::hgemm_col_nt_64x64_8x8::generate(*AS_SRC{IO});

close(AS_SRC);

my @outlines = <AS_OUT>;
my @errlines = <AS_ERR>;
print "STDOUT:\n", @outlines, "\n";
print "STDERR:\n", @errlines, "\n";

waitpid( $pid, 0 );
