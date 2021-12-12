import torch
from torch import nn
from torch.nn import Embedding,Linear
from torch.optim import Adam
from text2sequence import Word2Sequence
from text2dataloader import get_dataloader, ws, max_len
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.embedding = Embedding(len(ws), 100)  # 字典大小 和 词向量的大小
        self.fc = Linear(max_len * 100, 5)

    def forward(self, batch):
        """

        :param batch: [batch, max_len]
        :return:
        """
        x = self.embedding(batch)  # 返回[batchsize,maxlen,100]
        x = x.view([-1, max_len * 100])
        output = self.fc(x)
        return F.log_softmax(output,dim=-1)

model = EmbeddingModel()
optimizer = Adam(model.parameters(),lr=0.001)
def train(epoch):
    for index,(data,label) in enumerate(get_dataloader()):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,label)
        loss.backward()
        optimizer.step()
        print(loss.item())

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0.0
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            # 此时不会保存变化，没有grad_fn这个属性
            y_ = model(x)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("\n Test set : Average Loss: {:.4f},Accuracy:{}/{} ({:.0f}%)".format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)
    ))
