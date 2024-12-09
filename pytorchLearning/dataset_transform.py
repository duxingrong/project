"""
主要讲解一下torchvision中的数据集的下载和使用
"""
import torchvision
from torch.utils.tensorboard import SummaryWriter
import  matplotlib.pyplot as plt 


dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10",train=True,transform=dataset_transforms,download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10",train=False,transform=dataset_transforms,download=True)

print(test_set[0])
print(test_set.classes)

img, target = test_set[0]
print(type(img))
print(target)
print(test_set.classes[target])

"""
本来使用tensorboard的，但是觉得没有plt快
"""

# 将 Tensor 转换为 NumPy 数组并调整维度顺序 (HWC)
img = img.permute(1, 2, 0).numpy()
# 将数值范围从 [0, 1] 转换到 [0, 255]
img = (img * 255).astype('uint8')
plt.imshow(img)
plt.title(test_set.classes[target])  # 打印类别名
plt.show()
