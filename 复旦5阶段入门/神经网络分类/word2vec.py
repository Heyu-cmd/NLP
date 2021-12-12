from gensim.models import word2vec,Word2Vec
import logging
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from gensim.models import KeyedVectors
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def word_cut(sentence):
    """
    英文分词
    :param sentence: 句子
    :return: 单词的列表
    """
    tokenizer = RegexpTokenizer(r'\w+')  # 删除标点符号
    tokens = tokenizer.tokenize(sentence)
    return tokens

def word_embedding():
    File_name = "../机器学习分类/"
    df = pd.read_csv(File_name + "train.tsv", sep="\t")
    text_matrix = []
    for d in df.iloc[:, -2]:
        d2 = word_cut(d)
        text_matrix.append(d2)
    """
    sentences：可以是一个List，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建
    sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    size：是指输出的词的向量维数，默认为100
    window： 为训练的窗口大小，8表示每个词考虑前8个词与后8个词
    alpha: 是学习速率
    seed：用于随机数发生器。与初始化词向量有关
    min_count:词频少于min_count次数的单词会被丢弃掉, 默认值为5
    sample:  表示 采样的阈值，如果一个词在训练样本中出现的频率越大，那么就越会被采样。默认为1e-3，范围是(0,1e-5)
    workers:参数控制训练的并行数
    hs:是否采用层次softmax
    negative：如果>0,则会采用负采样，用于设置多少个noise words
    cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（default）则采用均值
    hashfxn： hash函数来初始化权重。默认使用python的hash函数 
    iter： 迭代次数，默认为5。
    
    """
    model = word2vec.Word2Vec(text_matrix, sg=1, vector_size=100, window=5, min_count=1, workers=4,epochs=10)
    model.save('./word2vec.model')
word_embedding()
model = KeyedVectors.load("./word2vec.model")
print(model.wv['the'])
sims = model.wv.most_similar('good',topn=10)
print(sims)
