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


def train(train_iter, vocab_size, lr, hidden_size, device='cpu'):
    def initialize_params(device):
        w_hx = torch.normal(0, 0.01, size=(vocab_size, hidden_size), device=device)
        w_hh = torch.normal(0, 0.01, size=(hidden_size, hidden_size), device=device)
        b_h = torch.zeros(hidden_size, device=device)
        w_ho = torch.normal(0, 0.01, size=(hidden_size, vocab_size), device=device)
        b_o = torch.zeros(vocab_size, device=device)
        H = torch.zeros((batch_size, hidden_size), device=device)
        return w_hx, w_hh, b_h, w_ho, b_o, H
    
    w_hx, w_hh, b_h, w_ho, b_o, H_temp = initialize_params()
    for X in train_iter:
        Y, H_temp = NET(X, H_temp, w_hx, w_hh, b_h, w_ho, b_o)



if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    F.one_hot(torch.tensor([0, 2]), len(vocab))
