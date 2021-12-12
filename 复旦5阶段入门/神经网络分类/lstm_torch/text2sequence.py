"""
构建字典
文本转换为序列，及翻译
未知字符的替换
短文本的填充

"""
import pickle
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import re
class Word2Sequence:
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }

        self.count = {}  # 统计词频

    def fit(self, sentence):
        """

        :param sentence: [word1,word2,word3,...] word:str
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_feature=None):
        """
        生成词典
        :param min:最小出现次数
        :param max: 最大出现次数
        :param max_feature: 一共保留多少的词语
        :return:
        """
        # 删除count中词频小于min的word
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        # 删除count中词频小于max的word
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        # 保留max_feature个word
        if max_feature is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_feature]
            self.count = dict(temp)
        for word in self.count:
            self.dict[word] = len(self.dict)

        # 得到一个反转的字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        句子转换为序列
        :param sentence:[word1,word2,...] word:str
        :return: list:[int,int,int...]
        """
        sentence_len = len(sentence)
        if max_len is not None:
            if max_len > sentence_len:
                sentence = sentence + [self.PAD_TAG] * (max_len - sentence_len)
            if max_len < sentence_len:
                sentence = sentence[:max_len]
        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, sequence):
        """
        序列转换为句子
        :param sequence: [int,int,int]
        :return:
        """
        return [self.inverse_dict.get(index) for index in sequence]
    def __len__(self):
        return len(self.dict)
def tokenize(content):
    filter = ['\t', '\n', ',', '-', '``', '\'\'', '\.', '\(', '\)']
    content = re.sub("|".join(filter), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

if __name__ == '__main__':
    ws = Word2Sequence()
    df = pd.read_csv("/Users/heyup/PycharmProjects/NLP_exec/复旦5阶段入门/机器学习分类/train.tsv",sep="\t")
    text = df.Phrase.tolist()
    for sentence in tqdm(text):
        data = tokenize(sentence)
        ws.fit(data)

    ws.build_vocab(min=10)
    pickle.dump(ws, open("./model/ws.pkl", 'wb'))
    print(len(ws.dict))
