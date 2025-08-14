import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1024, 1024, device=device)
for _ in range(1000):
    x = torch.matmul(x, x)
