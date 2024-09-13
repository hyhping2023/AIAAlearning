import torch
from d2l.torch import load_data_fashion_mnist

def SVMLoss(w, b, x, penalty, y, cuda='cpu'):
    # print('w shape: ', w.shape, 'b shape: ', b.shape, 'x shape: ', x.shape, 'y shape: ', y.shape)
    x = x.to(cuda)
    w = w.to(cuda)
    b = b.to(cuda)
    y = y.to(cuda)

    result = x@w + b
    # result = result.to(cuda)
    # print('y: ', y)
    
    model = torch.ones(result.shape)
    model = model.to(cuda)
    model *= result[0, y]
    penaltyMatrix = torch.ones(result.shape)*penalty
    penaltyMatrix = penaltyMatrix.to(cuda)
    substract = model - result + penaltyMatrix
    substract[substract < 0.] = 0.
    
    loss = (torch.sum(substract, dim=1) - penalty) / (substract.shape[1] - 1)
    # loss = loss.sum()
    # print(f'sum: {loss.sum()} mean: {loss.mean()}')
    loss = loss.mean()
    # print('model ', model.shape)
    # print('penaltyMatrix:', penaltyMatrix.shape)
    # print('result:', result.shape)
    # print('substract:', substract.shape)
    # print('model:', model.shape)
    # print('loss:', loss.shape)
    return loss

num_inputs = 784
num_outputs = 10

def train(device= 'cpu'):
    lr = 1
    penalty = 0.1
    batch_size = 1024
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    w = torch.normal(0, 0.0001, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros((1, num_outputs), requires_grad=True)
    best_score = 10000000
    for _ in range(20):
        count = 0
        for X, y in train_iter:
            # print('shape', X.shape)
            loss = SVMLoss(w, b, X.reshape(-1, num_inputs), penalty, y.reshape(-1, 1), device)
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad / batch_size
                b -= lr * b.grad / batch_size
                w.grad.zero_()
                b.grad.zero_()
            count += 1
            # if count > 1000:
            #     break
        # print(w, b)
        print(loss)
        if loss < best_score:
            results = (w, b)
            best_score = loss
    print(results)
    return results

def test(w, b):
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    results = []
    for X, y in train_iter:
        result = X.reshape(-1, num_inputs)@w + b
        # print(result.shape)
        y_predict = torch.argmax(result, dim=1)
        # print(f'Predict: {y_predict}, Actual: {y}')
        results.extend((y_predict == y).tolist())
    print(f'Accuracy: {sum(results)/len(results)}')

if __name__ == '__main__':
    w,b = train('cpu')
    test(w, b)

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

    

