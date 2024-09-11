import torch
from d2l.torch import load_data_fashion_mnist

def SVMLoss(w, b, x, penalty, y):
    # print('w shape: ', w.shape, 'b shape: ', b.shape, 'x shape: ', x.shape, 'y shape: ', y.shape)
    result = x@w + b
    # print('y: ', y)
    # print('result:', result)
    model = torch.ones(result.shape)*result[0, y]
    # print('model:', model)
    substract = result - model + torch.ones(result.shape)*penalty
    substract[substract < 0.] = 0.
    # print('substract:', substract)
    loss = (torch.sum(substract, dim=1) - penalty) / (substract.shape[1] - 1)
    # print(loss)
    return loss

num_inputs = 784
num_outputs = 10

def train():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可
    lr = 0.01
    penalty = 0.01
    train_iter, test_iter = load_data_fashion_mnist(1)
    w = torch.normal(0, 0.0001, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros((1, num_outputs), requires_grad=True)
    for _ in range(20):
        count = 0
        for X, y in train_iter:
            # print('shape', X.shape)
            loss = SVMLoss(w, b, X.reshape(-1, num_inputs), penalty, y)
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
            count += 1
            if count > 10:
                break
        # print(w, b)
        print(loss)

if __name__ == '__main__':
    # x = torch.tensor([[1., 1.],[2., 2.], [3., 3.]])
    # w = torch.tensor([[1.], [2.]], requires_grad=True)
    # print(x)
    # for _ in range(20):
    #     loss = SVMLoss(w=w, b=torch.tensor([[1.],[2.],[3.]]), x=x, penalty=0.5, y=0)
    #     loss.backward()
    #     # print(x.grad)
    #     with torch.no_grad():
    #         grad = w.grad
    #         w -=  0.1 * grad
    #         w.grad.zero_()
    #     if (_+1) % 2 == 0:
    #         print(f'epoch {_+1}: loss = {loss}')
    # print(w, loss)

    train()

