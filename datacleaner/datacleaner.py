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

# six.PY2 - python2
# six.PY3 - python3
import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

# config file
from . import config
from .drawer import Drawer
from .cls import SVM

import os
# jieba - chinese word break
import jieba
import logging
import random
from collections import Counter

logger = logging.getLogger('DataCleaner')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class DataCleaner(object):

    """ init
    @param: input    - input file
    @param: output   - out folder
    @usage: 
    dc = DataCleaner(
        input='/path/to/file', 
        output='output/to/folder',
        tokenizer=None,
        filter=None
    )
    
    """
    def __init__(self, input=None, output=None, tokenizer=None, filter=None):
        # user input parameters
        self.input = input
        self.output = output
        self.tokenizer = self._tokenizer if tokenizer == None else tokenizer
        self.filter = self._filter if filter == None else filter

        # load stopwords
        self._stopwords = self._load_stopwords()
        # word counter
        self._word_counter = Counter()
        # folders  
        self._folder_dict = {}
        # files
        self._file_dict = {}
        self._hotword_dict = {}

        self._clf = None
        self._clf_dict = {}
        self._clf_datas = []
        self._clf_datas_raw = []
        self._clf_train_data_dict = {}

        # start to run
        self._setup()

    """ stopword
    @param: w    - word list/tuple
    @param: f    - file path

    """
    def stopword(self, w=[], f=None):
        logger.info('set user stopword')
        words = set()
        fwords = set()
        if len(w) > 0:
            words = set(w)
        if f != None:
            if os.path.exists(f):
                fwords = set(self._load_stopwords(f))

        self._stopwords = set(self._stopwords) | words | fwords


    """ hotword
    @param: k_cloud - k words showed in word cloud
    @param: k_bar   - k words showed in bar
    """
    def hotword(self, k_cloud=100, k_bar=20, min_len=2, reverse=True, bg_cloud=None):
        self._hotword_dict = {
            'k_cloud': k_cloud,
            'k_bar': k_bar,
            'min_len': min_len,
            'reverse': reverse,
            'bg': bg_cloud
        }

    
    """ classify
    @param: datas   -  dict 
    @param: kernel  -  classify kernel , default svm
    @param: confidence
    """
    def classify(self, datas, kernel='svm', confidence=0.6):
        self._clf_dict = {
            'datas': datas,
            'kernel': kernel,
            'confidence': confidence,
        }


    """ run
    @desc: start to run
    """
    def run(self):
        self._process()
        if len(self._hotword_dict) > 0:
            self._hotword()
        if len(self._clf_dict) > 0:
            self._classify()

    """ _hotword
    @desc: start to process hotword
    """
    def _hotword(self):
        logger.info('process hotword')

        self._word_counter = dict(sorted(self._word_counter.items(), key=lambda x: x[1], reverse=self._hotword_dict['reverse']))

        if len(self._word_counter) == 0:
            raise ValueError('no any words!')

        prefix = '_reverse' if not self._hotword_dict['reverse'] else ''
        word_cloud_file = os.path.join(self._folder_dict['hotword'], 'word_cloud{}.png'.format(prefix))
        word_bar_file = os.path.join(self._folder_dict['hotword'], 'word_bar{}.png'.format(prefix))
        word_txt_file = os.path.join(self._folder_dict['hotword'], 'word_counter{}.txt'.format(prefix))

        word_counter_dict = dict(filter(lambda x: len(x[0]) >= self._hotword_dict['min_len'], self._word_counter.items()))
        word_counter = [(k,v) for k, v in word_counter_dict.items()]
        Drawer.word_cloud(word_counter_dict, self._hotword_dict['k_cloud'], word_cloud_file, bg=self._hotword_dict['bg'])
        Drawer.word_bar(word_counter, self._hotword_dict['k_bar'], word_bar_file)
        with open(word_txt_file, 'w') as f:
            for k, v in self._word_counter.items():
                f.writelines('{}\t{}\n'.format(k, v)) 

        logger.info('process hotword done!')

    """ _hotword
    @desc: start to process classify
    """
    def _classify(self):
        logger.info('process classify')

        total = 0

        for label, data in self._clf_train_data_dict.items():
            total += len(data)
        avg = total // len(self._clf_train_data_dict.keys())

        train_datas = []
        train_labels =  []
        out_datas = {}
        for label, data in self._clf_train_data_dict.items():
            if len(data) < avg:
                train_data = data + data*(avg - len(data))
            else:
                train_data = data
            train_datas.extend(train_data[:avg])
            train_labels.extend([label] * avg)
            out_datas[label] = []

        # shuffle dataset
        dataset = list(zip(train_datas, train_labels))
        random.shuffle(dataset)
        train_datas, train_labels = zip(*dataset)        

        if self._clf_dict['kernel'] == 'SVM':
            self._clf = SVM()
            self._clf.train(train_datas, train_labels)
            preds = self._clf.predict(self._clf_datas)
            for pred, data in zip(preds, self._clf_datas_raw):
                if pred[0][1] > self._clf_dict['confidence']:
                    out_datas[pred[0][0]].append(data)
        else:
            out_datas = dict(zip(train_labels, train_datas))

        for label, datas in out_datas.items():
            with open(os.path.join(self._folder_dict['classify'], label), 'w') as f:
                for data in datas:
                    f.writelines(data + '\n')

        logger.info('process classify done!')



    """ _process
    @desc: process input file
    """
    def _process(self):
        logger.info('process data')

        if len(self._hotword_dict) + len(self._clf_dict) == 0:
            return

        if len(self._clf_dict) > 0:
            for label, data in self._clf_dict['datas'].items():
                self._clf_train_data_dict[label] = []

        with open(self.input, 'r') as f, open(self._file_dict['token_file'], 'w') as f1:
            for line in f:
                if six.PY2:
                    line = line.decode('utf8')

                line = line.strip().replace(' ','')
                line_filter = self.filter(line.lower())
                if len(line_filter) == 0:
                    continue

                line_tok = self.tokenizer(line_filter)

                seg_line = ' '.join(line_tok)
                f1.writelines(seg_line + '\n')

                if len(self._clf_dict) > 0:
                    self._clf_datas.append(line_tok)
                    self._clf_datas_raw.append(line)

                    comms = []
                    for label, data in self._clf_dict['datas'].items():
                        comm = set(data) & set(line_tok)
                        comms.append((label, len(comm)))
                    comms = sorted(comms, key=lambda x: x[1], reverse=True)
                    if comms[0][1] > 0:
                        self._clf_train_data_dict[comms[0][0]].append(line_tok)

                for tok in line_tok:
                    if tok not in self._stopwords:
                        self._word_counter[tok] += 1

            logger.info('process data done!')


    """ _setup
    """
    def _setup(self):
        logger.info('init...')

        def check():
            # check input exists 
            if self.input != None:
                if not os.path.exists(self.input):
                    raise ValueError('input file/folder is not found!')

            # check out folder
            if self.output == None:
                raise ValueError('output folder is required!')
        # check
        check()

        # create out folder
        self.output = self.output.rstrip(os.sep)
        # make output folder
        self._make_dirs(self.output)

        self._folder_dict = {
            'runtime': os.path.join(self.output, 'runtime'),
            'hotword': os.path.join(self.output, 'hotword'),
            'classify': os.path.join(self.output, 'classify'),
        }
        # make dirs
        {self._make_dirs(d) for k, d in self._folder_dict.items()}

        input_filename = self.input.rstrip(os.sep).split(os.sep)[-1]
        self._file_dict['token_file'] = os.path.join(self._folder_dict['runtime'], '{}.token'.format(input_filename))
        
    

    """ _load_stopwords
    """
    def _load_stopwords(self, path=config.stopwords_path):
        logger.info('load stopwords')

        words = set()
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    words.add(line.strip())
        return list(words)


    """ _make_dirs
    """
    def _make_dirs(self, d):
        try:
            if six.PY3:
                os.makedirs(d, exist_ok=True)
                logger.info('make dir: {}'.format(d))
            else:
                if not os.path.exists(d):
                    os.makedirs(d)
        except Exception as e:
            raise ValueError('failed to create out folder, please sure that you have access to make dirs')

    """ _filter
    """
    def _filter(self, string):
        return string

    """ _tokenizer
    """
    def _tokenizer(self, string):
        return jieba.lcut(string)

logger.info('Done !')