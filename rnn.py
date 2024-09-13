import torch
import math
from d2l import torch as d2l
from torch.nn import functional as F

def NET(X, H, w_hx, w_hh, b_h, w_ho, b_o):
    H_temp = torch.tanh(torch.tanh(X@w_hx) + H@w_hh + b_h)
    Y = H_temp@w_ho + b_o
    return Y, H_temp

def LOSS(y_hat, y):
    pass


def train(train_iter, vocab_size, lr, hidden_size, batch_size, device='cpu'):
    def initialize_params(device, vocab_size, hidden_size, batch_size):
        w_hx = torch.normal(0, 0.01, size=(vocab_size, hidden_size), requires_grad=True, device=device)
        w_hh = torch.normal(0, 0.01, size=(hidden_size, hidden_size), requires_grad=True, device=device)
        b_h = torch.zeros(hidden_size, requires_grad=True, device=device)
        w_ho = torch.normal(0, 0.01, size=(hidden_size, vocab_size), requires_grad=True, device=device)
        b_o = torch.zeros(vocab_size, requires_grad=True, device=device)
        H = torch.zeros(batch_size, hidden_size, requires_grad=True, device=device)
        return w_hx, w_hh, b_h, w_ho, b_o, H
    
    w_hx, w_hh, b_h, w_ho, b_o, H_temp = initialize_params(device, vocab_size, hidden_size, batch_size)
    for X, y in train_iter:
        X = F.one_hot(X.T, vocab_size).type(torch.float32)
        # print(X.shape, y.shape)
        y_hat, H_temp = NET(X, H_temp, w_hx, w_hh, b_h, w_ho, b_o)
        # print(y.long())
    print(y_hat.shape, y.shape)



if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    F.one_hot(torch.tensor([0, 2]), len(vocab))
    # print(vocab)
    train(train_iter, len(vocab), lr=0.1, hidden_size=32, batch_size=32, device='cpu')
