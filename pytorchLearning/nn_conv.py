"""
稍微理解一下卷积的具体操作过程
"""
import torch 
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])


#改变input和kernel的shape,因为con2d接收4维
input = torch.reshape(input , (1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

print(input.shape)
print(kernel.shape)

#stride 步长  padding 填充
output = F.conv2d(input , kernel , stride=2 ,padding = 0)
print(output)

output2 = F.conv2d(input , kernel , stride=1, padding=1 )
print(output2)