import torch

x = torch.empty(5,3)

x = torch.rand(5,3)

x = torch.zeros(5,3)

x = torch.ones(5,3)
y = x.new_ones(2,2)

y = torch.randn_like(x,dtype=torch.float)


print(x)
print(y)

#加法
print(x+y) #dtype should be same

result = torch.rand(5,3)
torch.add(x,y,out=result)
print(result)

print(y.add(x))


#view 转变视图
print(x.view(15))

print(x.view(-1,5)) #-1表示自适应

#.item() if you have one item, you can use .item() to show the element
x = torch.rand(1)
print(x.item())

#numpy bridge

a = torch.rand(4,4)

b = a.numpy()
print(type(a),type(b))

#if we change a, b will also change
a.add_(1)
print("a:%s,b:%s"%(a,b))

#numpy to tensor
import numpy as np
a = np.ones(4)

b = torch.from_numpy(a)

np.add(a,b,out=a)
print(type(a),type(b))
print(a,b)