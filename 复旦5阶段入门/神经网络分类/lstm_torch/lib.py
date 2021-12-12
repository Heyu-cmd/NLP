import pickle
from text2sequence import Word2Sequence
import torch
ws = pickle.load(open("/Users/heyup/PycharmProjects/NLP_exec/复旦5阶段入门/神经网络分类/lstm_torch/model/ws.pkl", 'rb'))
max_len = 20  # 每个句子的单词数
word_dim = 200 # 每个单词的embedding维度
batch_size = 128 # 每批数据的样本数量
hiden_size = 5 # lstm的单元数量
num_layers = 2 # lstm的层数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
