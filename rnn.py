import torch
import math
from d2l import torch as d2l
from torch.nn import functional as F

def NET(X, H, w_hx, w_hh, b_h, w_ho, b_o):
    H_temp = torch.tanh(torch.tanh(X@w_hx) + H@w_hh + b_h)
    Y = H_temp@w_ho + b_o
    return Y, H_temp

def LOSS(y_hat, y):
    return -torch.log(torch.softmax(y_hat, dim=1)[range(len(y_hat)), y])

def grad_clipping(W, theta = 1):
    norm = torch.sqrt(sum(torch.sum((w.grad) ** 2) for w in W))
    if norm > theta:
        for w in W:
            w.grad *= theta / norm

def train(train_iter, vocab_size, lr, hidden_size, batch_size, device='cpu'):
    def initialize_params(device, vocab_size, hidden_size, batch_size):
        w_hx = torch.normal(0, 0.01, size=(vocab_size, hidden_size), requires_grad=True, device=device)
        w_hh = torch.normal(0, 0.01, size=(hidden_size, hidden_size), requires_grad=True, device=device)
        b_h = torch.zeros(hidden_size, requires_grad=True, device=device)
        w_ho = torch.normal(0, 0.01, size=(hidden_size, vocab_size), requires_grad=True, device=device)
        b_o = torch.zeros(vocab_size, requires_grad=True, device=device)
        H = torch.zeros(batch_size, hidden_size, device=device)
        return w_hx, w_hh, b_h, w_ho, b_o, H
    
    w_hx, w_hh, b_h, w_ho, b_o, H_temp = initialize_params(device, vocab_size, hidden_size, batch_size)
    
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        X = F.one_hot(X.T, vocab_size).type(torch.float32)
        # print(X.shape, y.shape)
        y_hat, H_temp = NET(X, H_temp.detach(), w_hx, w_hh, b_h, w_ho, b_o)

        '''
        学习心得
        在共用隐藏层时要用detach()来吧隐藏层从计算图中分离，不然会出现已经backward过的问题
        '''
        y_hat = y_hat.to(device)
        l = LOSS(y_hat, y.long()).mean()
        l.backward()
        grad_clipping([w_hx, w_hh, b_h, w_ho, b_o], 1)
        # print(w_hh.grad)
        # print(w_hh)
        with torch.no_grad():  
            w_hx -= lr * w_hx.grad
            w_hh -= lr * w_hh.grad
            b_h -= lr * b_h.grad
            w_ho -= lr * w_ho.grad
            b_o -= lr * b_o.grad
            w_hx.grad.zero_()
            w_hh.grad.zero_()
            b_h.grad.zero_()
            w_ho.grad.zero_()
            b_o.grad.zero_()
    return w_hx, w_hh, b_h, w_ho, b_o, l

def predict(prefix, num_preds, w_hx, w_hh, b_h, w_ho, b_o, vocab, device='cpu'):
    w_hx, w_hh, b_h, w_ho, b_o = w_hx.to(device), w_hh.to(device), b_h.to(device), w_ho.to(device), b_o.to(device)
    # 预热期
    H = torch.zeros(1, w_hh.shape[1], device=device)
    for i in range(len(prefix) - 1):
        _, H = NET(F.one_hot(torch.tensor([vocab[prefix[i]]], device=device), len(vocab)).type(torch.float32), H, w_hx, w_hh, b_h, w_ho, b_o)
    # 预测
    pred = [vocab[prefix[0]]]
    for _ in range(num_preds):
        _, H = NET(F.one_hot(torch.tensor([pred[-1]], device=device), len(vocab)).type(torch.float32), H, w_hx, w_hh, b_h, w_ho, b_o)
        pred.append(int(torch.argmax(_, axis=1)))
    print(pred)
    return ''.join([vocab.idx_to_token[i] for i in pred])


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    F.one_hot(torch.tensor([0, 2]), len(vocab))
    # print(vocab)
    for _ in range(500): # epoches
        w_hx, w_hh, b_h, w_ho, b_o, l = train(train_iter, len(vocab), lr=150, hidden_size=32, batch_size=batch_size, device='cpu')
        if _ % 50 == 0:
            print(l)
    # print(len(vocab))
    print(predict('time', 100, w_hx, w_hh, b_h, w_ho, b_o, vocab, device='cpu'))