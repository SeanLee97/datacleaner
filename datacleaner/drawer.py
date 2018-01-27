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

import numpy as np
from scipy.misc import imread
from . import config
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud, ImageColorGenerator

class Drawer(object):
    
    @staticmethod
    def word_cloud(word_counter, max_words, output, bg=None, is_show=True, font=None):
        bg =  config.drawer_bg if bg == None else bg 
        font = config.drawer_font if font == None else font
        bg_mask = imread(bg)

        wc = WordCloud(
            font_path=font,
            background_color='white',
            mask=bg_mask,
            max_words=max_words,
            max_font_size=60
        )

        wc.fit_words(word_counter)
        image_colors = ImageColorGenerator(bg_mask)
        wc.to_file(output)
        if is_show:
            plt.figure()
            plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
            plt.axis("off")
            plt.show()

    @staticmethod
    def word_bar(word_counter, max_words, output, title='', font=None):
        def autolabel(rects, ax):
            for rect in rects:
                width = rect.get_width()
                ax.text(1.03 * width, rect.get_y() + rect.get_height()/2.,  
                    '%d' % int(width),ha='center', va='center')

        fig, ax = plt.subplots()
        font = config.drawer_font if font == None else font
        font_handle = FontProperties(fname=font)
        words = []
        counts = []
        for idx, item in enumerate(word_counter):
            if idx > max_words:
                break
            words.append(item[0])
            counts.append(item[1]) 

        y_pos = np.arange(max_words)

        colors = ['#FA8072'] #这里是为了实现条状的渐变效果，以该色号为基本色实现渐变效果
        for i in range(len(words[:max_words]) - 1):
            colors.append('#FA' + str(int(colors[-1][3:]) - 1))

        rects = ax.barh(y_pos, counts[:max_words], align='center', color=colors)

        ax.set_yticks(np.arange(max_words))
        ax.set_yticklabels(words[:max_words],fontproperties=font_handle)
        ax.invert_yaxis()
        ax.set_title(title, fontproperties=font_handle, fontsize=17)
        ax.set_xlabel(u"word times",fontproperties=font_handle)

        autolabel(rects, ax)

        # save pic
        plt.savefig(output)
        plt.show()    