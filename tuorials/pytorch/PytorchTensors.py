"""
Pytorch Tutorial

Reference:
https://www.youtube.com/watch?v=IC0_FRiX-sw&t=12s&ab_channel=PyTorch
"""

import torch

div = "\n" + "-" * 50 + "\n"
print(div)

print("Creating a tensor with torch.zeros")
print("z = torch.zeros(5, 3)")
z = torch.zeros(5, 3)
print(z)
print("z.dtype shows the data type of the tensor elements")
print(z.dtype)
print(div)

print("Creating a tensor with torch.ones and specify the data type")
print("z = torch.ones(5, 3), dtype=torch.int16")
z = torch.ones((5, 3), dtype=torch.int16)
print(z)
print("z.dtype shows the data type of the tensor elements")
print(z.dtype)
print(div)