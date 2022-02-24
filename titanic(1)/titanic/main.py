import torch
import torch.nn as nn
from datapre import *
from config import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class TitanicPredictor(nn.Module):
    def __init__(self, args):
        super(TitanicPredictor, self).__init__()
        self.device = args.device
        self.args = args

        self.PclassEmbed = nn.Embedding(4, args.emb_size)
        self.SexEmbed = nn.Embedding(2, args.emb_size)
        self.SibSpEmbed = nn.Embedding(9, args.emb_size)
        self.ParchEmbed = nn.Embedding(7, args.emb_size)
        self.EmbarkEmbed = nn.Embedding(3, args.emb_size)

        self.input_size = 2 + 5 * args.emb_size
        self.hidden_nodes = args.hidden_nodes
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_nodes),
            nn.Sigmoid(),
            # nn.Linear(self.hidden_nodes, self.hidden_nodes),
            # nn.Sigmoid(),
            nn.Linear(self.hidden_nodes, 1),
            nn.Sigmoid()
        )

        self.batch_size = args.batch_size

    def get_new_batch(self, batch):
        # batch: (B, 7)
        cols = torch.split(batch, 1, dim=1)
        new_pclass = self.PclassEmbed(cols[0].long()).view(-1, self.args.emb_size)
        new_sex = self.SexEmbed(cols[1].long()).view(-1, self.args.emb_size)
        new_sib = self.SibSpEmbed(cols[3].long()).view(-1, self.args.emb_size)
        new_parch = self.ParchEmbed(cols[4].long()).view(-1, self.args.emb_size)
        new_embark = self.EmbarkEmbed(cols[6].long()).view(-1, self.args.emb_size)
        new_batch = torch.cat([new_pclass, new_sex, cols[2], new_sib, new_parch, cols[5], new_embark], dim=1)

        return new_batch.to(torch.float32)

    def forward(self, batch):
        #batch: (B, 7)
        input = self.get_new_batch(batch.to(args.device))

        output = self.net(input)
        return output.squeeze()

def make_batches(data):
    pass_num = data.shape[0]
    batch_num = int(pass_num / args.batch_size) + 1
    batches = []

    for i in range(batch_num):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        if end > pass_num:
            end = pass_num
        batches.append((start, end))
    return batches

#from read_data to dataset
#return a dataset including train and test and batches
def divide():
    data, lable = read_data('./data/train.csv')
    test_num=72 #approximately 10%

    data_temp=torch.split(data,[data.shape[0]-test_num,test_num],dim=0)
    train_data=data_temp[0]
    test_data=data_temp[1]

    lable_temp=torch.split(lable,[lable.shape[0]-test_num,test_num],dim=0)
    train_lable=lable_temp[0].float()
    test_lable=lable_temp[1].float()

    dataset={'train':{'data':train_data,'lable':train_lable,'batches':make_batches(train_data)},
             'test':{'data':test_data,'lable':test_lable,'batches':make_batches(test_data)}}
    return dataset

def train(args):
    losses = []

    TP = TitanicPredictor(args).to(args.device)
    loss_func = nn.BCELoss().to(args.device)
    optim = torch.optim.SGD(TP.parameters(),lr=args.lr)
    dataset=divide()
    train_data=dataset['train']['data']
    train_lable=dataset['train']['lable']
    batches=dataset['train']['batches']

    TP.train()
    losses = []
    for epoch in range(args.epochs):
        totalloss = 0
        start_time = datetime.now()
        for start, end in batches:
            input = train_data[start:end]
            lable = train_lable[start:end]
            output = TP(input)

            loss = loss_func(output, lable.to(args.device))
            # losses.append(loss.item())
            totalloss += loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            # print(f"epoch-{epoch}, iter-{end}, loss-{loss}")

        losses.append(totalloss.item() / len(batches))
        if epoch % 500 == 0:
            print(f"epoch-{epoch}, time-{datetime.now() - start_time}, "
                  f"loss-{totalloss / len(batches)}")  # replace batch_num with len(bathces)
            test(args, epoch)
        if epoch % 5000 == 0 and epoch>0 :
            now=datetime.now()
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': TP.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss,
                }
                ,'./checkpoint/cp-epoch-'+str(epoch)++'-time-'+str(now)+ '.pth.tar'
            )
            print("model saved to './checkpoint/cp-epoch-'"+str(epoch)+'-time-'+str(now)+ '.pth.tar')

    epoches = [i for i in range(args.epochs)]
    plt.plot(epoches, losses)
    plt.show()

def load_checkpoint(model, checkpoint_PATH, optimizer):
    #if checkpoint != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        return model, optimizer

class AccMetric:
    def __init__(self):
        self.match = 0
        self.total_n = 0

    def update(self, predict, lable):
        for i, p in enumerate(predict):
            if p == lable[i]:
                self.match += 1
        self.total_n += len(predict)

    def show(self):
        return self.match / self.total_n

def test(args, epoch):
    AM = AccMetric()
    TP = TitanicPredictor(args).to(args.device)
    dataset = divide()
    test_data = dataset['test']['data']
    test_lable = dataset['test']['lable']
    batches = dataset['test']['batches']
    for start, end in batches:
        input = test_data[start:end]
        lable = test_lable[start:end]
        with torch.no_grad():
            output = TP(input)

        predict = []
        for prob in list(output):
            if prob >= 0.5:
                predict.append(1)
            else:
                predict.append(0)

        AM.update(predict, lable)

    print(f"epoch-{epoch}, acc-{AM.show()}")

if __name__ == "__main__":
    train(args)
    # test(args)