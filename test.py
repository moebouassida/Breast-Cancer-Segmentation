import torch
print(torch.__version__)
print(torch.version.cuda)   # should print a version like 12.1, not None
print(torch.cuda.is_available())