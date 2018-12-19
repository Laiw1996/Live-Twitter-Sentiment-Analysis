import pickle
# import flask
import data
import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import *
import config

# app = flask.Flask(__name__)

# model = torch.load("model/rnn.pkl")


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
        Variable(sequence_tensor), length)
    out = model(packed_sequences)
    # out = F.softmax(out, dim=1)
    _, out_label = torch.max(out, 1)

    return out_label

def predict(sentence):
    # # GET all the sentences from the csv file
    # df = pd.read_csv(data_folder, header=None)
    # df.columns = ['sentence']
    # sentences = df['sentence']
    # # sentences = df.iloc[:, 1]
    #
    vocab = data.Vocabulary()
    data.build_vocab(vocab)
    #
    # raw_data = data.sentimentDatasetPredict(vocab, sentences)
    # print("create vocab")
    # vocab = data.Vocabulary()
    # data.build_vocab(vocab)
    #
    # print("loading data")
    # data_loader = DataLoader(raw_data, batch_size=128, shuffle=True,
    #                          collate_fn=data.collate_fn_predict)

    model1 = RNNClassifier(nembedding=config.DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          num_layer=config.NUM_LAYER,
                          dropout=config.drop_out,
                          vocab_size=vocab.n_words,
                          use_pretrain=True,
                          embed_matrix=vocab.vector,
                          embed_freeze=False,
                          label_size=3)

    model2 = CNNClassifier(nembedding=config.DIM,
                                            vocab_size=vocab.n_words,
                                            kernel_num=3,
                                            kernel_sizes=config.kernel_sizes,
                                            label_size=3,
                                            dropout=config.drop_out,
                                            use_pretrain=True,
                                            embed_matrix=vocab.vector,
                                            embed_freeze=False)
    state_dict1 = torch.load("model/rnn.pth")
    model1.load_state_dict(state_dict1)

#    state_dict2 = torch.load("model/cnn1.pkl")
#    model2.load_state_dict(state_dict2)

    out_rnn = output_sentiment(sentence, model1, vocab).numpy()
#    out_cnn = output_sentiment(sentence, model2, vocab).numpy()

#    prediction = []
#
#    if(out_cnn[0] > out_rnn[0]):
#        #prediction.append(out_cnn[0])
#        result = out_cnn[0]
#    else:
#        #prediction.append(out_rnn[0])
#        result = out_rnn[0]
    return out_rnn[0]


#    print(prediction)
#    print("done")


    # # model.eval()
    # prediction = []
    # for batch in data_loader:
    #     out1 = model1(batch['sentence'])
    #     out2 = model2(batch['sentence'])
    #     _, out_label1 = torch.max(out1, 1)
    #     _, out_label2 = torch.max(out2, 1)
    #     # use RNN and CNN in the same time, use the largest label as out predicted label
    #     for i in range(len(out_label1.numpy())):
    #         use_rnn = out_label1.numpy()
    #         use_cnn = out_label2.numpy()
    #         if(use_cnn[i] >= use_rnn[i]):
    #             prediction.append(use_cnn[i])
    #         else:
    #             prediction.append(use_rnn[i])



if __name__ == "__main__":
    predict("rt this scoop be accurate per wh official with direct knowledge")