from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

img_path = r"C:\Users\19390\Desktop\project\pytorchLearning\dataset\train\bees\36900412_92b81831ad.jpg"
img_PIL = Image.open(img_path)
print(type(img_PIL))

#ToTensor
trans_totensor = transforms.ToTensor()
tensor_img = trans_totensor(img_PIL)
writer.add_image("Tensor",tensor_img ,1)

#Normalize 
print(tensor_img[0][0][0])#取第0层第0行第0列像素
trans_norm = transforms.Normalize([0.5,0.4,0.4],[0.4,0.5,0.6])
img_norm = trans_norm(tensor_img)
writer.add_image("Normalize",img_norm,1)
print(img_norm[0][0][0])

#Resize 
print(img_PIL.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img_PIL)
print(img_resize.size)
#然后将PIL变成tensor
img_resize = trans_totensor(img_resize)
writer.add_image("resize",img_resize,1)


#Compose函数,就是一个组合的作用
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img_PIL)
writer.add_image("Resize_2" , img_resize_2,1)

#RandomCrop,就是把图片按照你给的尺寸随机提取
trans_random = transforms.RandomCrop((30,40))
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_random = trans_compose_2(img_PIL)
    writer.add_image("Random_img",img_random,i)




writer.close()