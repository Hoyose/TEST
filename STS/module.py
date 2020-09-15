# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.nums_head = args.nums_head

        self.bn = nn.BatchNorm1d(self.embedding_size)

        self.lstm1 = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * self.nums_head, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * self.nums_head),
            nn.Linear(self.hidden_size * self.nums_head, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, 2),
            nn.Softmax(dim=-1)
        )

    @staticmethod
    def sub_mul(s1, s2):
        mul = s1 * s2
        sub = s1 - s2

        return torch.cat([sub, mul], -1)

    @staticmethod
    def apply_multiple(s):
        """

        :param s: [batch_size, sentence_length, hidden_size]
        :return: [batch_size, 2 * hidden_size]
        """
        p1 = F.avg_pool1d(s.transpose(1, 2), s.size(1)).squeeze(-1)
        p2 = F.max_pool1d(s.transpose(1, 2), s.size(1)).squeeze(-1)

        return torch.cat([p1, p2], dim=1)

    def forward(self, *input):
        s1 = input[0]
        s2 = input[1]

        # [batch_size, sentence_length, embedding_size] => [batch_size, sentence_length, hidden_size]
        o1, _ = self.lstm1(s1)
        o2, _ = self.lstm1(s2)

        # Aggregate
        # input: [batch_size, sentence_length, hidden_size]
        # output: [batch_size, 2 * hidden_size]
        s1_rep = self.apply_multiple(o1)
        s2_rep = self.apply_multiple(o2)

        # Classifier
        x = torch.cat([s1_rep, s2_rep], dim=1)
        sim = self.fc(x)
        return sim

