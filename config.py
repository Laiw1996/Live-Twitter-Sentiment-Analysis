import torch

# file_name = "./SAD.csv"
# file_name = "./sentiment-analysis-dataset.csv"
file_name = "data/cleaned_data.csv"
# data config
MIN_FREQ = 10
MAX_VOCAB_SIZE = 40000
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3


# model config
DIM = 300  # embedding dims
HIDDEN_SIZE = 128
NUM_LAYER = 2
drop_out = 0.5
EPOCH = 20
lr = 0.1
weight_decay = 1e-10

# cnn config
kernel_sizes = (3, 4, 5)
