import os, time, shutil, argparse

import data
import model
import config
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# # Parse arguments and prepare program
# parser = argparse.ArgumentParser(description='Twitter live analysis')
#
# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to .pth file checkpoint (default: none)')
# parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (overridden if loading from checkpoint)')
# parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='size of mini-batch (default: 16)')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='learning rate at start of training')
# parser.add_argument('--weight-decay', '--wd', default=1e-10, type=float, metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='use this flag to validate without training')
# parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')

# we have two model for training data: RNN, CNN
def output_sentiment(sentence, model, vocab):
    "using model to distinguish whether a sentence is neg or pos"
    sentence = sentence.split()  # list
    length = np.array([max(len(sentence), config.kernel_sizes[-1])])  # length must be greater than max kernel size
    sequence = [0] * length[0]
    for i, word in enumerate(sentence):
        sequence[i] = vocab.get_idx(word)

    sequence_tensor = torch.LongTensor(sequence).unsqueeze(1)
    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
        Variable(sequence_tensor).cuda(), length)
    out = model(packed_sequences)
    out = F.softmax(out, dim=1)
    return out


def train(batch, model, optimizer, criterion):
    '''Train model on data in train_loader for a single batch'''

    model.train()
    optimizer.zero_grad()
    x, y = batch['sentence'], batch['sentiment']
    y_pred = model(x)
    loss = criterion(y_pred, y.squeeze())
    loss.backward()
    optimizer.step()
    return loss.data


def test(batch, model):
    '''Test model on data in train_loader for a single batch'''
    text, label = batch['sentence'], batch['sentiment']
    out = model(text)
    _, out_label = torch.max(out, 1)
    accuracy = (torch.sum(out_label == label).cpu().data.numpy() / len(label))  # have to convert bytetensor-->numpy
    return accuracy


def run(lr, weight_decay):
    # lr = 0.1
    # weight_decay = 1e-10

    print("starting...")
    # prepare data
    csv_dataset = pd.read_csv(config.file_name,
                              header=None)  # csv_file format: dataframe
    print("Loading dataset.")
    vocab = data.Vocabulary()
    data.build_vocab(vocab)  # build vocabulary

    print("Building vocabulary.")
    train_data = data.sentimentDataset(vocab, csv_dataset,
                                       train_size=config.TRAIN_RATIO,
                                       test_size=config.TEST_RATIO, train=True)
    test_data = data.sentimentDataset(vocab, csv_dataset, train=False)

    train_loader = DataLoader(train_data,
                                  batch_size=config.TRAIN_BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.TEST_BATCH_SIZE,
                                 shuffle=True, collate_fn=data.collate_fn)

    model_classifier1 = model.RNNClassifier(nembedding=config.DIM,
                                            hidden_size=config.HIDDEN_SIZE,
                                            num_layer=config.NUM_LAYER,
                                            dropout=config.drop_out,
                                            vocab_size=vocab.n_words,
                                            use_pretrain=True,
                                            embed_matrix=vocab.vector,
                                            embed_freeze=False,
                                            label_size=3)

    model_classifier2 = model.CNNClassifier(nembedding=config.DIM,
                                            vocab_size=vocab.n_words,
                                            kernel_num=3,
                                            kernel_sizes=config.kernel_sizes,
                                            label_size=3,
                                            dropout=config.drop_out,
                                            use_pretrain=True,
                                            embed_matrix=vocab.vector,
                                            embed_freeze=False)

    # optimizer = optim.Adam(model_classifier2.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model_classifier2.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.EPOCH):
        print("epoch", epoch)
        for batch in train_loader:
            print(train(batch, model_classifier2, optimizer, criterion))

    print("saving model")
    if isinstance(model_classifier1, model.RNNClassifier):
        torch.save(model_classifier1.state_dict(), "rnn.pth")
    if isinstance(model_classifier2, model.CNNClassifier):
        torch.save(model_classifier2.state_dict(), "cnn.pth")


    print("testing...")
    model_classifier2.eval()
    sum = 0
    cnt = 0
    for batch in test_loader:
        sum += test(batch, model_classifier2)
        cnt += 1
    print("accuracy:", sum / cnt)



if __name__ == "__main__":
    # global args, best_losses, use_gpu
    # args = parser.parse_args()
    # print('Arguments: {}'.format(args))


    # use_loaded_model = input("load model?")
    # if use_loaded_model == "yes":
    #     if isinstance(model_classifier, model.RNNClassifier):
    #         print("loading RNN...")
    #         model_classifier.load_state_dict(torch.load("rnn.pkl"))
    #     if isinstance(model_classifier, model.CNNClassifier):
    #         print("loading CNN...")
    #         model_classifier.load_state_dict(torch.load("cnn.pkl", map_location=lambda storage, loc: storage))
    lr = 0.1
    weight_decay = 1e-10

    run(lr, weight_decay)
