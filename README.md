![datacleaner.png](https://github.com/SeanLee97/datacleaner/blob/master/docs/datacleaner.png)

# datacleaner
* 词频统计，生成词云图，词频统计图
* 初步分类器

分类器思想：为了从原始数据中较快速的对文本进行分类，在此先用关键词获取少批量数据，将获取到的批量数据放进分类器中（目前仅封装了svm）训练，得到模型后在去跑原始数据，从而得到初步的分类数据。当然分类后的数据难免存在噪音，所以还得继续对数据进行处理。**datacleaner仅作为第一层的数据处理工具**

## 环境
* python2.7 / python3.5+

## 实例
```python
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
    'football': ['足球', '西甲', '欧冠', '世界杯'],
}, kernel='svm', confidence=0.8)

"""开始执行处理"""
dc.run()
```

## 结果
![word_cloud.png](https://github.com/SeanLee97/datacleaner/blob/master/docs/word_cloud.png)

![word_bar.png](https://github.com/SeanLee97/datacleaner/blob/master/docs/word_bar.png)

## TODO-LIST
* [ ] word2vec cluster
