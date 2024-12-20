"""
损失函数
"""
import torch 
from torch import nn

inputs = torch.tensor([1,2,3] ,dtype=torch.float32)
targets = torch.tensor([1,2,5] , dtype=torch.float32)

inputs = torch.reshape(inputs , (1,1,1,3))
targets = torch.reshape(targets , (1,1,1,3))


#这个是计算结果和预期的误差值,单纯的是减法的绝对值总和或者平均
loss = nn.L1Loss(reduction='sum') #reduction="mean"或者'sum'

result = loss(inputs,targets)

print(result)

#平方差
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs,targets)

print(result_mse)

#再来一个交叉熵损失函数
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)