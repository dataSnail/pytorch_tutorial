import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MINST=False
if_use_gpu = False #是否使用GPU计算，必须有N卡

train_data =torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST
)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i'%train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile = True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, #the number of channel
                out_channels = 16, #输出的channel数量
                kernel_size = 5, #过滤器（核）数量
                stride=1, #步长
                padding=2, #扩展
            ),
            nn.ReLU(), #非线性变换
            
            nn.MaxPool2d(kernel_size=2), #池化层
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

if __name__ == '__main__':

    cnn = CNN()
    # print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()


    for epoch in range(EPOCH):
        for step, (x,y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            if if_use_gpu:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            output = cnn(b_x)
            # print(b_y)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if step % 100 == 0:  
                print('Epoch:', epoch, '|Step:', step,  
                  '|train loss:%.4f'%loss.item()) 
    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')

    # cnn = cnn.cpu()
    # test_output = cnn(test_x)  
    # pred_y = torch.max(test_output, 1)[1].data.squeeze()
    # accuracy = sum(pred_y == test_y) / test_y.size(0) 
    # print('Test Acc: %.4f'%accuracy)
