import torch
import torch.nn as nn
from datapre import *
from config import *
import numpy as np
from datetime import datetime

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
            nn.Linear(self.hidden_nodes, self.hidden_nodes),
            nn.Sigmoid(),
            nn.Linear(self.hidden_nodes, 2),
            nn.Softmax(dim=1)
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
        input = self.get_new_batch(batch)

        output = self.net(input)
        return output

def train(args):
    TP = TitanicPredictor(args).to(args.device)
    loss_func = nn.CrossEntropyLoss().to(args.device)
    optim = torch.optim.SGD(TP.parameters(),lr=args.lr)

    train_data, train_lable = read_data('./data/train.csv')
    pass_num = train_data.shape[0]
    batch_num = int(pass_num / args.batch_size) + 1
    batches = []

    for i in range(batch_num-1):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        if end > pass_num:
            end = pass_num
        batches.append((start, end))

    TP.train()
    losses = []
    for epoch in range(args.epochs):
        totalloss = 0
        start_time = datetime.now()
        for start, end in batches:
            input = train_data[start:end]
            lable = train_lable[start:end]
            output = TP(input)

            loss = loss_func(output, lable)
            losses.append(loss)
            totalloss += loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            # print(f"epoch-{epoch}, iter-{end}, loss-{loss}")
        print(f"epoch-{epoch}, time-{datetime.now()-start_time}, loss-{totalloss/batch_num}")





if __name__ == "__main__":
    train(args)