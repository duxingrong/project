"""
验证训练好的模型的效果
"""
from PIL import Image
import torchvision
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_dog_path = "pytorchLearning/dog.png"

image_dog = Image.open(img_dog_path)
print(image_dog)

image_feiji_path = "pytorchLearning/feiji.png"
image_feiji = Image.open(image_feiji_path)
print(image_feiji)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(),
])

# 处理输入
image_dog  = transform(image_dog)
print(image_dog.shape)
image_dog = torch.reshape(image_dog,(1,3,32,32))
image_dog = image_dog.to(device)

image_feiji  = transform(image_feiji)
print(image_feiji.shape)
image_feiji = torch.reshape(image_feiji,(1,3,32,32))
image_feiji = image_feiji.to(device)

# 模型验证
model=torch.load("tudui_20.pth")
print(model)
model.eval()
with torch.no_grad():
    output_dog = model(image_dog)
    output_feiji = model(image_feiji)
    print(output_dog.argmax(1)) #为target[5] : 'dog' 说明训练后的模型还行
    print(output_feiji.argmax(1))# 为target[0]: "bord' 成鸟了