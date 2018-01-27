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


import os
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

class SVM(object):
    def __init__(self):
        clf = SVC(kernel='linear', probability=True)
        self.estimator = GridSearchCV(clf, param_grid={'C': [1., 1.3, 1.5, 2., 2.3, 2.5, 3., 3.3, 3.5]})
        self.tfidf_vector = TfidfVectorizer(binary=False)

    def train(self, datas, labels):
        if type(datas[0]) == list:
            datas = [' '.join(line) for line in datas]
        data_vec = self.tfidf_vector.fit_transform(datas)
        self.estimator.fit(data_vec, labels)

    def predict(self, datas, k=1):
        preds = []
        if type(datas[0]) == list:
            data_vec = [self.tfidf_vector.transform([' '.join(line)]) for line in datas]
        else:
            data_vec = [self.tfidf_vector.transform([line]) for line in datas]
        for vec in data_vec:
            probas = self.estimator.best_estimator_.predict_proba(vec)
            pred = zip(self.estimator.best_estimator_.classes_, probas[0].tolist())
            pred = list(sorted(pred, key=lambda x: x[1], reverse=True))
            if k < len(pred):
                pred = pred[:k]
            preds.append(pred)
        return preds