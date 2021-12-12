from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import pickle
from tqdm import tqdm
import torch
from text2sequence import Word2Sequence
from lib import ws, max_len
import lib

# 1. prepare dataset and dataloader
"""
每个batch里，句子的长度不同
单词要变成数字
"""
ws = pickle.load(open("/Users/heyup/PycharmProjects/NLP_exec/复旦5阶段入门/神经网络分类/lstm_torch/model/ws.pkl", 'rb'))


def tokenize(content):
    filter = ['\t', '\n', ',', '-', '``', '\'\'', '\.', '\(', '\)']
    content = re.sub("|".join(filter), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


class FudanDataset(Dataset):
    def __init__(self, train=True):
        self.path = r"/Users/heyup/PycharmProjects/NLP_exec/复旦5阶段入门/机器学习分类"
        self.train = train
        if train:
            self.data_path = self.path + r"/train.tsv"
        else:
            self.data_path = self.path + r"/test.tsv"
        self.df = pd.read_csv(self.data_path, sep="\t")

    def __getitem__(self, item):
        raw_data = self.df.iloc[item]
        label = raw_data['Sentiment']
        data = raw_data['Phrase']
        data = tokenize(data)

        return data,label

    def __len__(self):
        return self.df.shape[0]


def collate(batch):
    """

    :param batch: 多个getitem的结果组成的元组([tokens,label][tokens,label][]...)
    :return:
    """
    ret = zip(*batch)
    text, label = list(ret)
    text = [ws.transform(i, lib.max_len) for i in text]
    text = torch.LongTensor(text)
    label = torch.LongTensor(label)
    return text, label


def get_dataloader(train=True):
    fudan_dataset = FudanDataset(train=train)
    data_loader = DataLoader(fudan_dataset, batch_size=lib.batch_size, shuffle=True, collate_fn=collate)
    return data_loader


if __name__ == '__main__':
    dataloader = get_dataloader()
    for index, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass
