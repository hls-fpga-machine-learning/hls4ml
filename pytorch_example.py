import torch
import torch.nn as nn
import torch.nn.functional as F

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# model = Net()

# class LayerLinearRegression(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Instead of our custom parameters, we use a Linear layer with single input and single output
#         self.linear = nn.Linear(1, 1)
                
#     def forward(self, x):
#         # Now it only takes a call to the layer to make predictions
#         return self.linear(x)

# model = LayerLinearRegression()


class MyModuleConvRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,3,3)
        
    def forward(self, x):
        y1 = self.conv(x)
        y = torch.relu(y1)
        y = y + y1
        y = torch.relu(y)
        return y


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

class MyModuleSoftMax(nn.Module):
    def __init__(self):
        super(MyModuleSoftMax, self).__init__()
        self.fc1 = nn.Linear(3,4)  
        self.fc2 = nn.Linear(4,2)
        self.sm  = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.softmax(self.fc2(x))
        x = self.sm(self.fc2(x))
        return


model = MyModuleConvRelu()


class MyModuleBatchNorm(nn.Module):
    def __init__(self):
        super(MyModuleBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,
                               kernel_size=5,
                               stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.dense1 = nn.Linear(in_features=320, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(-1, 320) #reshape
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.relu(self.dense2(x))
        #return F.softmax(x)
        return x


model = MyModuleConvRelu()



print(model)


print ("content of model:")
for layer_name, pytorch_layer in model.named_modules():
    print(layer_name, pytorch_layer.__class__.__name__)
print ("----------------------------------------------")

from torch.fx import symbolic_trace
traced_model = symbolic_trace(model)  
print(traced_model.graph)

config = config_from_pytorch_model(model)
#hls_model = convert_from_pytorch_model(model, (None, 1,30,30), hls_config = config ) #for MyModuleBatchNorm
#hls_model = convert_from_pytorch_model(model, (None, 3), hls_config = config ) #MyModuleSoftMax
hls_model = convert_from_pytorch_model(model, (None, 3, 5, 5), hls_config = config ) #MyModuleConvRelu

