import torch
import torch.nn as nn

class CNN_Model(nn.Module):
        def __init__(self):
            super(CNN_Model, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d( #(3, 224, 224)
                    in_channels=3,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=2
                    ),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )       # (16, 112, 112)
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=8,
                    kernel_size=5,
                    stride=1,
                    padding=2
                    ),
                nn.ReLU(),
                nn.MaxPool2d(2)
                ) # (8, 56, 56)
            self.out = nn.Linear(8 * 56 * 56, 2)

        def forward(self, input):
            x = self.conv1(input)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            output = self.out(x)
            return output, x

def Detect(Image):
    cnn = CNN_Model()
    cnn = torch.load('D:\python\Petdetect\pets.pt')
    Output, _ = cnn(Image)
    predict = torch.max(Output, dim=1)[1].data.numpy()
    return predict
