# code: utf-8

import os
import json
import torch
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer


class DataSet:
    def __init__(self, s1, s2, label):
        self.index_in_epoch = 0
        self.s1 = s1
        self.s2 = s2
        self.label = label
        self.example_nums = len(label)
        self.epochs_completed = 0

    def next_batch(self, batch_size, shuffled=False):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch >= self.example_nums:
            # Finished epoch
            self.epochs_completed += 1
            # Start next epoch
            start = 0
            end = self.index_in_epoch - batch_size
            # print(str(end)+'-------------------')
            self.index_in_epoch = batch_size
            assert batch_size <= self.example_nums
            # Shuffle the data
            if shuffled:
                perm = np.arange(self.example_nums)
                np.random.shuffle(perm)
                self.s1 = self.s1[perm]
                self.s2 = self.s2[perm]
                self.label = self.label[perm]
                # print(self.s1[start:self.index_in_epoch])
                return self.s1[start:self.index_in_epoch], self.s2[start:self.index_in_epoch], \
                    self.label[start:self.index_in_epoch]
            else:
                # print(self.s1[end:])
                return self.s1[end:], self.s2[end:], self.label[end:]
        else:
            # print(self.s1[start:self.index_in_epoch])
            return self.s1[start:self.index_in_epoch], self.s2[start:self.index_in_epoch], \
                self.label[start:self.index_in_epoch]


def padding(s1, s2):
    s1_padding = np.zeros([len(s1), 30], dtype=np.int)
    s2_padding = np.zeros([len(s1), 30], dtype=np.int)
    for index in range(len(s1)):
        if len(s1[index]) <= 30:
            s1_padding[index][:len(s1[index])] = s1[index][:]
        else:
            s1_padding[index][:] = s1[index][:30]
    for index in range(len(s2)):
        if len(s2[index]) <= 30:
            s2_padding[index][:len(s2[index])] = s2[index][:]
        else:
            s2_padding[index][:] = s2[index][:30]

    return s1_padding, s2_padding


def encode(batch_s1, batch_s2):
    """

    :param batch_s1: [batch_size, sentence_length]
    :param batch_s2: [batch_size, sentence_length]
    :return: [batch_size, sentence_length, embedding_size]
    """

    bert_model = BertModel.from_pretrained('E:/Projects/bert-pytorch/bert-base-chinese.tar.gz').to('cuda')
    bert_model.eval()
    batch_s1 = torch.Tensor(batch_s1).cuda(0).long()
    out1, _ = bert_model(batch_s1)
    # batch_s2 = torch.Tensor(batch_s2).cuda(0).long()
    # out2, _ = bert_model(batch_s2)
    print(out1[0].shape)


def read_data(file_path):
    s1 = []
    s2 = []
    label = []
    tokenizer = BertTokenizer.from_pretrained('E:/Projects/bert-pytorch/bert-base-chinese-vocab.txt')
    with open(file_path, encoding='utf-8') as read_file:
        lines = read_file.readlines()
        for line in lines:
            items = json.loads(line)
            marked_s1 = '[CLS] ' + items['sentence1'] + ' [SEP]'
            marked_s2 = '[CLS] ' + items['sentence2'] + ' [SEP]'
            s1_token = tokenizer.tokenize(marked_s1)
            s2_token = tokenizer.tokenize(marked_s2)
            s1_seq_ids = tokenizer.convert_tokens_to_ids(s1_token)
            s2_seq_ids = tokenizer.convert_tokens_to_ids(s2_token)
            s1.append(s1_seq_ids)
            s2.append(s2_seq_ids)
            label.append(items['label'])
    read_file.close()
    label = np.asarray(label, dtype=np.int)
    label = label.reshape([len(label), 1])
    # print(label.shape)
    return s1, s2, label


if __name__ == '__main__':
    dir_path = os.getcwd()

    train_file = "afqmc_public\\train.json"
    dev_file = "afqmc_public\\dev.json"
    test_file = "afqmc_public\\test.json"

    train_path = os.path.join(dir_path, train_file)

    read_s1, read_s2, read_label = read_data(train_path)

    padding_s1, padding_s2 = padding(read_s1, read_s2)

    AFQ_train = DataSet(padding_s1, padding_s2, read_label)
    for i in range(2200):
        train_s1, train_s2, train_label = AFQ_train.next_batch(16, True)
        encode(train_s1, train_s2)
