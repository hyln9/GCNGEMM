# hgemm_col_nt_64x64_8x8.pm
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

package kernels::hgemm_col_nt_64x64_8x8;

use strict;

my %s =
(
    gnumx     => 0,
    gnumy     => 1,
    gidx      => 12,
    gidy      => 13,
    aBuf      => 14,
    bBuf      => 16,
    cBuf      => 18,
    lda       => 20,
    ldb       => 21,
    ldc       => 22,
    m         => 23,
    n         => 24,
    k         => 25,
    a         => 26,
    b         => 27,
    aDesc     => 28,
    bDesc     => 32,
    cDesc     => 36,
    ldai4     => 20,
    ldbi4     => 21,
    ldai      => 40,
    ldbi      => 41,
    ldai8     => 42,
    ldbi8     => 43,
    sizeA     => 44,
    sizeB     => 45,
    i         => 46,
    end       => 47,
    ldci4     => 22,
    ldci      => 48,
    ldci8     => 49,
    sizeC     => 50,
    ldci29    => 51,
    temp      => 52,
    mi4       => 23,
    mi        => 53,
    ni4       => 54,
    ni        => 55,
    magic     => 56,
    shift     => 57
);

my %v =
(
    C         => 0,
    A0        => 64,
    B0        => 72,
    A1        => 80,
    B1        => 88,
    loadA     => 96,
    loadB     => 104,
    lidxo     => 0,
    lidyo     => 1,
    lidx      => 112,
    lidy      => 113,
    lid4      => 114,
    lid16     => 115,
    zd        => 116,
    zi        => 116,
    trackA    => 120,
    ofstA     => 121,
    trackB    => 122,
    ofstB     => 123,
    readAs    => 124,
    readBs    => 125,
    writeS    => 126,
    trackC    => 120,
    ofstC     => 121,
    lidy4     => 116,
    temp      => 117,
    temp1     => 118,
    temp2     => 119
);

my %pat =
(
    ldArg32   => sub {return join' ','s_buffer_load_dword',"s$_[0],s[8:11],@{[$_[1]*0x10]}"},
    ldArg64   => sub {return join' ','s_buffer_load_dwordx2',"s[$_[0]:@{[$_[0]+1]}],",'s[8:11],',"@{[$_[1]*0x10]}"},
    ldDesc    => sub {return join' ','s_load_dwordx4',"s[$_[0]:@{[$_[0]+3]}],",'s[2:3],',"@{[0x180+$_[1]*0x20]}"},
    ld2Gnum   => sub {return join' ','s_buffer_load_dwordx2',"s[$_[0]:@{[$_[0]+1]}],",'s[4:7],','0x20'},
    ld2Gofst  => sub {return join' ','s_buffer_load_dwordx2',"s[$_[0]:@{[$_[0]+1]}],",'s[4:7],','0x60'},
    waitCnt   => sub {return join' ','s_waitcnt',(($_[0]=~s/l(\d+)/lgkmcnt($1)/r)=~s/v(\d+)/vmcnt($1)/r)},
    injBase   => sub {return join' ','s_add_u32',"s$_[0],","s$_[0],","s$_[1]\n",
                        ' 'x6,'s_addc_u32',"s@{[$_[0]+1]},","s@{[$_[0]+1]},","s@{[$_[1]+1]}"},
    injFmt    => sub {return join' ','s_or_b32',"s@{[$_[0]+1]},","s@{[$_[0]+1]},","0x80000\n",
                        ' 'x6,'s_and_b32',"s@{[$_[0]+1]},","s@{[$_[0]+1]},",'0x3fffffff'},
    injSize   => sub {return join' ','s_mov_b32',"s@{[$_[0]+2]},","s$_[1]"}
);

sub generate {
    my $AS_SRC = shift;
    print $AS_SRC <<EOS;
.kernel hgemm_col_nt_64x64_8x8
    .config
        .dims xy
        .cws 8, 8
        .uavid 11
        .localsize 4096
        .arg A, short*, global, const
        .arg B, short*, global, const
        .arg C, short*, global
        .arg lda, uint
        .arg ldb, uint
        .arg ldc, uint
        .arg m, uint
        .arg n, uint
        .arg k, uint
        .arg a, short
        .arg b, short
        .arg magic, int
        .arg shift, int
        .userdata PTR_UAV_TABLE,0,2,2
        .userdata IMM_CONST_BUFFER,0,4,4
        .userdata IMM_CONST_BUFFER,1,8,4
    .text
        s_mov_b32 s$s{temp}, 0
        s_mov_b32 m0, 4096
        @{[$pat{ldArg32}($s{magic},11)]}
        @{[$pat{ldArg32}($s{shift},12)]}
        @{[$pat{ld2Gnum}($s{gnumx})]}
        v_mov_b32 v$v{lidx}, v$v{lidxo}
        v_mov_b32 v$v{lidy}, v$v{lidyo}
        v_lshrrev_b32 v$v{lid4}, 1, v$v{lidy}
        v_lshlrev_b32 v$v{lid16}, 3, v$v{lidy}
        v_and_b32 v$v{lid16}, 15, v$v{lid16}
        v_or_b32 v$v{lid16}, v$v{lidx}, v$v{lid16}
        @{[$pat{waitCnt}('l0')]}
        @{[$pat{ldArg32}($s{lda},3)]}
        @{[$pat{ldArg32}($s{ldb},4)]}
        @{[$pat{ldArg32}($s{k},8)]}
        v_mov_b32 v$v{temp}, s$s{gidy}
        v_add_u32 v$v{temp}, vcc, s$s{gidx}, v$v{temp}
        v_mul_hi_u32 v$v{temp1}, s$s{magic}, v$v{temp}
        @{[$pat{waitCnt}('l0')]}
        @{[$pat{ldArg64}($s{aBuf},0)]}
        @{[$pat{ldDesc}($s{aDesc},0)]}
        v_lshrrev_b32 v$v{temp1}, s$s{shift}, v$v{temp1}
        v_mul_u32_u24 v$v{temp1}, v$v{temp1}, s$s{gnumy}
        v_sub_u32 v$v{temp}, vcc, v$v{temp}, v$v{temp1}
        @{[$pat{waitCnt}('l0')]}
        @{[$pat{ldArg64}($s{bBuf},1)]}
        @{[$pat{ldDesc}($s{bDesc},1)]}
        v_readfirstlane_b32 s$s{temp}, v$v{temp}
        s_cmp_lg_u32 s$s{magic}, 0
        s_cmov_b32 s$s{gidy}, s$s{temp}
        @{[$pat{waitCnt}('l0')]}
        @{[$pat{ldArg32}($s{m},6)]}
        @{[$pat{ldArg32}($s{n},7)]}
        s_lshr_b32 s$s{ldai}, s$s{ldai4}, 2
        s_lshl_b32 s$s{ldai8}, s$s{ldai4}, 1
        s_lshr_b32 s$s{ldbi}, s$s{ldbi4}, 2
        s_lshl_b32 s$s{ldbi8}, s$s{ldbi4}, 1
        s_mul_i32 s$s{sizeA}, s$s{k}, s$s{ldai8}
        s_mul_i32 s$s{sizeB}, s$s{k}, s$s{ldbi8}
        @{[$pat{waitCnt}('l0')]}
        @{[$pat{ldArg32}($s{ldc},5)]}
        @{[$pat{ldArg64}($s{cBuf},2)]}
        @{[$pat{ldDesc}($s{cDesc},2)]}
        @{[$pat{injBase}($s{aDesc},$s{aBuf})]}
        @{[$pat{injFmt}($s{aDesc})]}
        @{[$pat{injSize}($s{aDesc},$s{sizeA})]}
        @{[$pat{waitCnt}('l0')]}
        @{[$pat{ldArg32}($s{a},9)]}
        @{[$pat{injBase}($s{bDesc},$s{bBuf})]}
        @{[$pat{injFmt}($s{bDesc})]}
        @{[$pat{injSize}($s{bDesc},$s{sizeB})]}
        @{[$pat{waitCnt}('l0')]}
        s_lshr_b32 s$s{mi}, s$s{mi4}, 2
        s_lshr_b32 s$s{ni}, s$s{ni4}, 2
        v_mov_b32 v$v{temp1}, 0xfffffff
        v_mad_u32_u24 v$v{trackA}, s$s{gidx}, 0x10, v$v{lid16}
        v_cmp_lt_u32 vcc, v$v{trackA}, s$s{mi}
        v_cndmask_b32 v$v{trackA}, v$v{temp1}, v$v{trackA}, vcc
        v_mad_u32_u24 v$v{trackA}, s$s{ldai}, v$v{lid4}, v$v{trackA}
        v_mad_u32_u24 v$v{trackB}, s$s{gidy}, 0x10, v$v{lid16}
        v_cmp_lt_u32 vcc, v$v{trackB}, s$s{ni}
        v_cndmask_b32 v$v{trackB}, v$v{temp1}, v$v{trackB}, vcc
        v_mad_u32_u24 v$v{trackB}, s$s{ldbi}, v$v{lid4}, v$v{trackB}
        v_lshlrev_b32 v$v{ofstA}, 3, s$s{lda}
        v_lshlrev_b32 v$v{ofstB}, 3, s$s{ldb}
        v_mul_u32_u24 v$v{writeS}, 0x100, v$v{lid4}
        v_mad_u32_u24 v$v{writeS}, v$v{lid16}, 16, v$v{writeS}
        tbuffer_load_format_xyzw v[$v{loadA}:@{[$v{loadA}+3]}], v$v{trackA}, s[$s{aDesc}:@{[$s{aDesc}+3]}], 0 idxen format:[16_16_16_16,uint]
        tbuffer_load_format_xyzw v[$v{loadB}:@{[$v{loadB}+3]}], v$v{trackB}, s[$s{bDesc}:@{[$s{bDesc}+3]}], 0 idxen format:[16_16_16_16,uint]
        tbuffer_load_format_xyzw v[@{[$v{loadA}+4]}:@{[$v{loadA}+7]}], v[$v{trackA}:@{[$v{trackA}+1]}], s[$s{aDesc}:@{[$s{aDesc}+3]}], 0 idxen offen format:[16_16_16_16,uint]
        tbuffer_load_format_xyzw v[@{[$v{loadB}+4]}:@{[$v{loadB}+7]}], v[$v{trackB}:@{[$v{trackB}+1]}], s[$s{bDesc}:@{[$s{bDesc}+3]}], 0 idxen offen format:[16_16_16_16,uint]
        v_add_u32 v$v{trackA}, vcc, s$s{ldai8}, v$v{trackA}
        v_add_u32 v$v{trackB}, vcc, s$s{ldbi8}, v$v{trackB}
        v_lshlrev_b32 v$v{readAs}, 4, v$v{lidx}
        v_lshlrev_b32 v$v{readBs}, 4, v$v{lidy}
        v_add_u32 v$v{readBs}, vcc, 2048, v$v{readBs}
        v_mov_b32 v$v{zd}, 0
        v_mov_b32 v@{[$v{zd}+1]}, 0
        v_mov_b32 v@{[$v{zd}+2]}, 0
        v_mov_b32 v@{[$v{zd}+3]}, 0
        ds_write_b128 v$v{zi}, v[$v{zd}:@{[$v{zd}+3]}]
        s_mov_b32 s$s{i}, 0
        s_add_i32 s$s{end}, -1, s$s{k}
        s_lshr_b32 s$s{end}, s$s{end}, 3
        @{[$pat{waitCnt}('l0')]}
        s_lshr_b32 s$s{ldci}, s$s{ldci4}, 2
        s_lshl_b32 s$s{ldci8}, s$s{ldci4}, 1
        s_mul_i32 s$s{sizeC}, s$s{n}, s$s{ldci8}
        @{[$pat{injBase}($s{cDesc},$s{cBuf})]}
        @{[$pat{injFmt}($s{cDesc})]}
        @{[$pat{injSize}($s{cDesc},$s{sizeC})]}
EOS
    map {print $AS_SRC join(' ',' 'x7,'ds_read_b128',"v[@{[$v{C}+$_*4]}:@{[$v{C}+$_*4+3]}],","v$v{zi}\n")} 0..15;
    print $AS_SRC <<EOS;
        @{[$pat{waitCnt}('l0&v3')]}
        ds_write_b128 v$v{writeS}, v[$v{loadA}:@{[$v{loadA}+3]}]
        @{[$pat{waitCnt}('v2')]}
        ds_write_b128 v$v{writeS}, v[$v{loadB}:@{[$v{loadB}+3]}] offset:2048
        @{[$pat{waitCnt}('l0')]}
        ds_read_b128 v[$v{A0}:@{[$v{A0}+3]}], v$v{readAs}
        ds_read_b128 v[$v{B0}:@{[$v{B0}+3]}], v$v{readBs}
        ds_read_b128 v[@{[$v{A0}+4]}:@{[$v{A0}+7]}], v$v{readAs} offset:128
        ds_read_b128 v[@{[$v{B0}+4]}:@{[$v{B0}+7]}], v$v{readBs} offset:128
        @{[$pat{waitCnt}('l0')]}
LOOP_hgemm_col_nt_64x64_8x8:
EOS
    foreach my $i (0..7)
    {
        print $AS_SRC join(' 'x7,'ds_read_b128',"v[@{[$v{A1}+$i%2*($v{A0}-$v{A1})]}:@{[$v{A1}+$i%2*($v{A0}-$v{A1})+3]}],","v$v{readAs}","offset:@{[($i+1)%8*256]}\n");
        print $AS_SRC join(' 'x7,'ds_read_b128',"v[@{[$v{B1}+$i%2*($v{B0}-$v{B1})]}:@{[$v{B1}+$i%2*($v{B0}-$v{B1})+3]}],","v$v{readBs}","offset:@{[($i+1)%8*256]}\n");
        print $AS_SRC join(' 'x7,'ds_read_b128',"v[@{[$v{A1}+$i%2*($v{A0}-$v{A1})+4]}:@{[$v{A1}+$i%2*($v{A0}-$v{A1})+7]}],","v$v{readAs}","offset:@{[($i+1)%8*256+128]}\n");
        print $AS_SRC join(' 'x7,'ds_read_b128',"v[@{[$v{B1}+$i%2*($v{B0}-$v{B1})+4]}:@{[$v{B1}+$i%2*($v{B0}-$v{B1})+7]}],","v$v{readBs}","offset:@{[($i+1)%8*256+128]}\n");
        foreach my $j (0..7)
        {
            if ($i==0&&$j==3) {
                print $AS_SRC join(' ',' 'x7,'tbuffer_load_format_xyzw',"v[$v{loadA}:@{[$v{loadA}+3]}],","v$v{trackA},","s[$s{aDesc}:@{[$s{aDesc}+3]}],",'0','idxen',"format:[16_16_16_16,uint]\n");
            } elsif ($i==0&&$j==4) {
                print $AS_SRC join(' ',' 'x7,'tbuffer_load_format_xyzw',"v[$v{loadB}:@{[$v{loadB}+3]}],","v$v{trackB},","s[$s{bDesc}:@{[$s{bDesc}+3]}],",'0','idxen',"format:[16_16_16_16,uint]\n");
            } elsif ($i==4&&$j==3) {
                print $AS_SRC join(' ',' 'x7,'tbuffer_load_format_xyzw',"v[@{[$v{loadA}+4]}:@{[$v{loadA}+7]}],","v[$v{trackA}:@{[$v{trackA}+1]}],","s[$s{aDesc}:@{[$s{aDesc}+3]}],",'0','idxen','offen',"format:[16_16_16_16,uint]\n");
            } elsif ($i==4&&$j==4) {
                print $AS_SRC join(' ',' 'x7,'tbuffer_load_format_xyzw',"v[@{[$v{loadB}+4]}:@{[$v{loadB}+7]}],","v[$v{trackB}:@{[$v{trackB}+1]}],","s[$s{bDesc}:@{[$s{bDesc}+3]}],",'0','idxen','offen',"format:[16_16_16_16,uint]\n");
                print $AS_SRC join(' ',' 'x7,'v_add_u32',"v$v{trackA},",'vcc,',"s$s{ldai8},","v$v{trackA}\n");
                print $AS_SRC join(' ',' 'x7,'v_add_u32',"v$v{trackB},",'vcc,',"s$s{ldbi8},","v$v{trackB}\n");
            } elsif ($i==6&&$j==3) {
                print $AS_SRC join(' ',' 'x7,"@{[$pat{waitCnt}('v3')]}\n",' 'x6,'ds_write_b128',"v$v{writeS},","v[$v{loadA}:@{[$v{loadA}+3]}]\n");
                print $AS_SRC join(' ',' 'x7,'s_add_u32',"s$s{i},",'1,',"s$s{i}\n");
            } elsif ($i==6&&$j==4) {
                print $AS_SRC join(' ',' 'x7,"@{[$pat{waitCnt}('v2')]}\n",' 'x6,'ds_write_b128',"v$v{writeS},","v[$v{loadB}:@{[$v{loadB}+3]}]","offset:2048\n");
                print $AS_SRC join(' ',' 'x7,'s_cmp_gt_u32',"s$s{i},","s$s{end}\n");
            } elsif ($i==2&&$j==3) {
                print $AS_SRC join(' ',' 'x7,"@{[$pat{waitCnt}('v3')]}\n",' 'x6,'ds_write_b128',"v$v{writeS},","v[@{[$v{loadA}+4]}:@{[$v{loadA}+7]}]","offset:1024\n");
            } elsif ($i==2&&$j==4) {
                print $AS_SRC join(' ',' 'x7,"@{[$pat{waitCnt}('v2')]}\n",' 'x6,'ds_write_b128',"v$v{writeS},","v[@{[$v{loadB}+4]}:@{[$v{loadB}+7]}]","offset:3072\n")
            }
            foreach my $k (0..7)
            {
                print $AS_SRC join(' ',' 'x7,'v_mac_f16',"v@{[$v{C}+$j*8+$k]},","v@{[$v{A0}+$i%2*($v{A1}-$v{A0})+$k]},","v@{[$v{B0}+$i%2*($v{B1}-$v{B0})+$j]}\n");
            }
        }
    }
    print $AS_SRC <<EOS;
        s_cbranch_scc0 LOOP_hgemm_col_nt_64x64_8x8
        v_mov_b32 v$v{temp1}, 0xfffffff
        v_mov_b32 v$v{temp2}, 64
        v_lshlrev_b32 v$v{lidy4}, 2, v$v{lidy}
        v_mad_u32_u24 v$v{trackC}, s$s{gidx}, 16, v$v{lidx}
        v_add_u32 v$v{temp}, vcc, 8, v$v{trackC}
        v_cmp_lt_u32 vcc, v$v{trackC}, s$s{mi}
        v_cndmask_b32 v$v{trackC}, v$v{temp1}, v$v{trackC}, vcc
        v_cmp_lt_u32 vcc, v$v{temp}, s$s{mi}
        v_cndmask_b32 v$v{ofstC}, v$v{temp1}, v$v{temp2}, vcc
        v_mad_u32_u24 v$v{temp}, s$s{gidy}, 64, v$v{lidy4}
        v_mad_u32_u24 v$v{trackC}, v$v{temp}, s$s{ldci}, v$v{trackC}
        s_mul_i32 s$s{ldci29}, 29, s$s{ldci}
EOS
    foreach my $j (0..7)
    {
        foreach my $k (0..3)
        {
            print $AS_SRC join(' ',' 'x7,'v_mul_f16',"v@{[$v{C}+$j*8+$k]},","v@{[$v{C}+$j*8+$k]},","s$s{a}\n");
        }
        print $AS_SRC join(' ',' 'x7,'tbuffer_store_format_xyzw',"v[@{[$v{C}+$j*8]}:@{[$v{C}+$j*8+3]}],","v$v{trackC},","s[$s{cDesc}:@{[$s{cDesc}+3]}],",'0','idxen','glc',"format:[16_16_16_16,uint]\n");
        foreach my $k (4..7)
        {
            print $AS_SRC join(' ',' 'x7,'v_mul_f16',"v@{[$v{C}+$j*8+$k]},","v@{[$v{C}+$j*8+$k]},","s$s{a}\n");
        }
        print $AS_SRC join(' ',' 'x7,'tbuffer_store_format_xyzw',"v[@{[$v{C}+$j*8+4]}:@{[$v{C}+$j*8+7]}],","v[$v{trackC}:@{[$v{trackC}+1]}],","s[$s{cDesc}:@{[$s{cDesc}+3]}],",'0','idxen','offen','glc',"format:[16_16_16_16,uint]\n");
        print $AS_SRC join(' ',' 'x7,'v_add_u32',"v$v{trackC},",'vcc,',"s@{[$j==3?$s{ldci29}:$s{ldci}]},","v$v{trackC}\n");
    }
    print $AS_SRC ' 'x8 . "s_endpgm\n";
}
