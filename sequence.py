import torch
from matplotlib import pyplot as plt

def labelsGenerate(num, step = 0.01, function=lambda x: torch.sin(x), noise = 0.01):
    X = step * torch.arange(1, num + 1, dtype=torch.float32)
    labels = function(X) + torch.normal(0, noise, size=X.shape)
    # plt.plot(X, labels)
    return X, labels

def dataLoader(samNum, labels, backstep = 1,batch_size = 16, train = 0.6):
    allNum = len(labels)
    features = torch.zeros((allNum - samNum, samNum))
    for i in range(samNum):
        features[:, i] = labels[i: allNum - samNum + i]
    Y_hat = labels[samNum:].reshape((-1, 1))
    current = 0
    train_iter, test_iter = [], []
    allNum -= samNum
    train_num = int(allNum * train)
    while (train_num - current) // batch_size > 0:
        X = torch.cat([features[current: current + batch_size]], dim=1)
        y = torch.cat([Y_hat[current: current + batch_size]], dim=1)
        current += batch_size
        train_iter.append((X, y))
    if train_num - current > 0:    
        X = torch.cat([features[current: train_num]], dim=1)
        y = torch.cat([Y_hat[current: train_num]], dim=1)
        current = train_num
        train_iter.append((X, y))
    
    # For test iter
    allNum += samNum
    test_features = torch.zeros((allNum - numSam - backstep + 1, numSam + backstep))
    for i in range(numSam):
        test_features[:, i] = labels[i: i + allNum - numSam - backstep + 1]
            
    return train_iter, test_features, train_num

def Net(W1, W2, X):
    X1 = X @ W1 
    X1[X1 < 0.] = 0.0
    y_hat = X1 @ W2
    return y_hat

def grad_clipping(W, theta = 1):
    norm = torch.sqrt(sum(torch.sum((w.grad) ** 2) for w in W))
    if norm > theta:
        for w in W:
            w.grad *= theta / norm

def train(train_iter, lr, samNum, epochs = 20, device= 'cpu'):
    W1 = torch.normal(0, 0.01, size=(samNum, 10), requires_grad=True)
    W2 = torch.normal(0, 0.01, size=(10, 1), requires_grad=True)
    for epoch in range(epochs):
        for X, y in train_iter:
            y_hat = Net(W1, W2, X)
            Loss = (y_hat - y) ** 2 / 2
            Loss.sum().backward()
            with torch.no_grad():
                grad_clipping([W1, W2], 1)
                W1 -= lr * W1.grad
                W2 -= lr * W2.grad
                W1.grad.zero_()
                W2.grad.zero_()
        if epoch % 10 == 0:
            print(f'epoch: {epoch + 1}, loss: {Loss.sum():f}')
    return W1, W2

def predict(test_iter, W1, W2, numSam, backstep):
    for i in range(numSam, backstep + numSam):
        test_iter[:, i] = Net(W1, W2, test_iter[:, i - numSam: i]).reshape(-1)
        
    return test_iter[:, numSam + backstep - 1].reshape(-1, 1)


if __name__ == "__main__":
    allNum = 1000
    X, labels = labelsGenerate(allNum, noise=0.2, function=lambda x: torch.exp(x)/(1 + torch.exp(x)) + torch.sin(x))
    numSam = 4
    lr = 0.01
    batch_size = 2
    backstep = 8
    train_iter, test_iter, train_num = dataLoader(numSam, labels, batch_size = batch_size
                                                  , backstep=backstep, train=0.6)

    W1, W2 = train(train_iter, lr, numSam, epochs = 20, device= 'cpu')
    
    print(test_iter.shape)
    y_hat = predict(test_iter, W1, W2, numSam, backstep)
    y_train = torch.cat([_[1] for _ in train_iter], dim=0)
    y = labels.reshape(-1, 1)

    # change the size of the figure
    plt.rcParams['figure.figsize'] = (12, 6)

    argument = 1
    print(y_hat.shape, train_num)
    y = y[numSam + backstep - 1:, :]
    print(y.shape)
    # print(y_hat.shape, y_train.shape, X[y_train.shape[0] - argument+ numSam:].shape)

    plt.plot(X[numSam + backstep - 1:], y_hat.detach().numpy(), label='Predict', linestyle='--')
    plt.plot(X[numSam + backstep - 1:], y.detach().numpy(), label='True', linewidth=0.2, color='r')
    plt.legend()
    plt.show()