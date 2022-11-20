import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
trainPath = "D:\python\Petsource\CatDog\Train"
testPath = "D:\python\Petsource\CatDog\Test"


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

def Load_Data():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
        ])
    train_data = torchvision.datasets.ImageFolder(trainPath, transform = transforms)
    test_data = torchvision.datasets.ImageFolder(testPath, transform = transforms)
    # print(train_data.class_to_idx
    #train_size = int(0.7 * len(train_data))
    #test_size = len(train_data) - train_size
    #train_data, test_data = Data.random_split(train_data, [train_size, test_size])
    train_loader = Data.DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size = BATCH_SIZE, shuffle=True)
    #for step, (b_x, b_y) in enumerate(train_loader):
    #    print(f"batchX:{b_x}")
    #    print(f"batchY:{b_y}")
    return train_loader, test_loader

def Train_Model():
    trainLoader, testLoader = Load_Data()
    #test_x = torch.unsqueeze(testLoader.test_data, dim=1).type(torch.FloatTensor)[:500]/255
    #print(testLoader[0])
    #test_y = testLoader.labels[:500]
    cnn = CNN_Model()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    # Train
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(trainLoader):
            output = cnn(b_x)[0]  
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                #test_output, _ = cnn(test_x)
                #pred_y = torch.max(test_output, 1)[1].data.numpy()
                #accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                #print('epoch:', epoch, '| train loss :%.4f'% loss.data.numpy(), '| test accuracy:%.2f' %accuracy)
                print('epoch:', epoch, '| train loss :%.4f'% loss.data.numpy())
    
    try:
        torch.save(cnn, r'D:\python\Petdetect\pets.pt')
        print('save')
    except:
        print('save failure')


if __name__ == "__main__":
    Train_Model()
    #Load_Data()
    