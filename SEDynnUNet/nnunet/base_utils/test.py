import torch
from monai.networks import one_hot
label = torch.Tensor([[[0,2,1],[0,1,2],[0,1,0]]])

print("label.shape", label.shape)
print(label)
target = one_hot(label, num_classes=3, dim=0)
print(target)

label2 = torch.argmax(target, 0) #从onehot转换回去
print(label2)