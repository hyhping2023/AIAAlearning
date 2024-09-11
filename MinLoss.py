import torch

n = 100
d = 2
x = torch.rand(n, d)
noise = torch.randn(n, 1) * 0.01
w = torch.tensor([[1.0], [-2.0]])
b = torch.tensor([[3.0]])
y = torch.matmul(x, w) + b + noise

# 增广矩阵
xa = torch.cat((x, torch.ones(n, 1)), dim=1)
# print(xa)
wa = torch.cat((w, b), dim=0)

print(xa.t()@xa)
wa_pred = (xa.t()@xa).inverse()@xa.t()@y
print(wa_pred)
