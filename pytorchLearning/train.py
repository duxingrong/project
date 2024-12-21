"""
完整的模型训练步骤,采用GPU训练
"""
import torchvision 
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader


# 准备数据集合
train_data = torchvision.datasets.CIFAR10(root="data" , transform=torchvision.transforms.ToTensor(),train=True,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="data",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为:{train_data_size}")
print(f"测试数据集的长度为:{test_data_size}")

# 利用DataLoader 来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型,模型一般单独保存
from model import * 
tudui = Tudui()
tudui = tudui.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01 #或者 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0 

# 训练的次数
epoch = 20 

# 添加tensorboard 
writer = SummaryWriter("logs")



for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")

    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs ,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step+=1 
        if total_train_step%100 == 0:
            print(f"训练次数:{total_train_step},loss:{loss.item()}")# 写item()是从tensor中取出值
            writer.add_scalar("train_loss",loss.item(),total_train_step)
        
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0 
    total_accuracy = 0 
    with torch.no_grad():
        for data in  test_dataloader:
            imgs ,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs ,targets)
            total_test_loss+= loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy += accuracy
    
    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step+=1 

    # torch.save(tudui,f"tudui_{i+1}.pth")
    print("模型已保存")

writer.close()