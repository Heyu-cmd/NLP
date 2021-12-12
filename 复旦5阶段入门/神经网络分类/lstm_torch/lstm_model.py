from text2sequence import Word2Sequence
from lib import ws, word_dim, batch_size, hiden_size, num_layers, max_len, device
from text2dataloader import get_dataloader
from torch import nn
from torch.nn import LSTM, Linear, Embedding
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

BATCH_SIZE = batch_size
WORD_DIM = word_dim
HIDEN_SIZE = hiden_size
NUM_LAYERS = num_layers
SENTENCE_LEN = max_len
DROPOUT = 0.1
DEVICE = device


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = Embedding(len(ws), WORD_DIM)
        self.lstm = LSTM(input_size=WORD_DIM, hidden_size=HIDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True
                         , dropout=DROPOUT, bidirectional=True)
        self.fc = Linear(HIDEN_SIZE * 2, 5)

    def forward(self, batch):
        x = self.embedding(batch)  # 返回[batch_size,Sentence_len,Word_dim]
        # x:[batch_size,sentence_len, 2*hiden_size],h_n:[num_Layers*directional_num),batch_size,hiden_size]
        x, (h_n, c_n) = self.lstm(x)
        output_forword = h_n[-2, :, :]  # 正向最后一次输出
        output_backword = h_n[-1, :, :]  # 反向最后一次输出
        output = torch.cat([output_backword, output_forword], dim=-1)  # [batch_size,hiden_size*2]

        out = self.fc(output)
        return F.log_softmax(out, dim=-1)


model = MyModel().to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.01)


def train(epoch):
    dataloader = get_dataloader(train=True)
    for inx, (input, label) in enumerate(dataloader):
        input = input.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if inx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
            print('Train Epoch : {} [{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, inx * len(input), len(dataloader.dataset),
                       100. * inx / len(dataloader), loss.item()
            ))


def test():
    model.eval()
    loss_list = []
    acc_list = []
    test_loader = get_dataloader(False)
    for index, (data, label) in enumerate(test_loader):
        with torch.no_grad():
            output = model(data)
            cur_loss = F.nll_loss(output, label)
            loss_list.append(cur_loss.item())

            # 计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(label).float().mean()
            acc_list.append(cur_acc.item())

    print(np.mean(loss_list), np.mean(acc_list))


for i in range(10):
    train(i)
