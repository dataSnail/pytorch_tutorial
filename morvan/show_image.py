import matplotlib.pyplot as plt
import numpy as np

# 图像显示函数
defimshow(img):
   img = img /2+0.5     # 非标准的（unnormalized）
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                               shuffle=True, num_workers=2)
# 得到一些随机图像
dataiter =iter(trainloader)images, labels = dataiter.next()

# 显示图像
imshow(torchvision.utils.make_grid(images))

# 打印类标
print(' '.join('%5s'% classes[labels[j]] for j in range(4)))