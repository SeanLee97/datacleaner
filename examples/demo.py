# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

"""MIT License

Copyright (c) 2018 Sean

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import sys
sys.path.append("..")

import jieba
from datacleaner import DataCleaner 

"""全角转半角"""
def strQ2B(string, codec="utf8"):
    try:
        ustring = string.decode(codec, "ignore")
    except:
        ustring = string
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        try:
            rstring += unichr(inside_code)
        except:
            rstring += chr(inside_code)
    return rstring
"""自定义过滤函数，将全角转半角"""
def filter_fc(string):
    return strQ2B(string) 

"""自定义分词处理函数，默认是结巴分词"""
def tokenizer_fc(string):
    return jieba.lcut(string)

"""datacleaner实例化对象"""
dc = DataCleaner(input='input.txt', output='./dc/', filter=filter_fc, tokenizer=tokenizer_fc)

"""自定义停用词
@param: w   - 停用词列表
@param: f   - 停用词所在文件,以行分割每个词
"""
dc.stopword(w=['中国'], f=None)

"""设置热词生成
@param: k_cloud   -  词云图显示的词个数
@param: k_bar     -  条形词频统计词的个数
@param: min_len   -  词的最小长度
@param: reverse   -  词频是否降序排列（由高到低）
@param: bg_cloud  -  词云图的背景图片（背景图地址）
"""
dc.hotword(k_cloud=100, k_bar=20, min_len=2, reverse=True, bg_cloud=None)

"""定义分类器
@param: datas      -  传入dict，key是类名，value是关键词列表
@param: kernel     -  分类器，默认svm
@param: confidence -  置信度
"""
dc.classify({
    'basketball': ['篮球', '乔丹'],
    'football': ['足球', 'football'],
}, kernel='svm', confidence=0.8)

"""开始执行处理"""
dc.run()
