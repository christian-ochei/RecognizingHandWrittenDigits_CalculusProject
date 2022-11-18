import torch
import torch.nn as nn
import torchvision


class LinearLayer(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearLayer, self).__init__()
        self.w = nn.Parameter(torch.randn([input_size,output_size]))
        self.b = nn.Parameter(torch.randn([output_size]))

    def forward(self,x):
        x = (x @ self.w) + self.b
        return x


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l0 = LinearLayer(784,784)
        self.l1 = LinearLayer(784,784)
        self.l2 = LinearLayer(784,784)
        self.l3 = LinearLayer(784,784)
        self.l4 = LinearLayer(784,784)
        self.l5 = LinearLayer(784,10)

    def forward(self,x):
        x = x.flatten(1)
        x = torch.sigmoid(self.l0(x))
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        x = torch.softmax(self.l5(x),dim=-1)
        return x

    def predict(self,x):
        pred = self(x)[0]
        return int(torch.argmax(pred))

def train():
    batch_size_train = 32
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    model = torch.load('MNISTModel6.pt')
    # model = MNISTModel()
    optim = torch.optim.Adam(model.parameters(),lr=0.0001)
    while True:
        for batch_idx, (train_data, train_targets) in enumerate(train_loader):
            # print(train_data[0])
            z = model(train_data)

            y = torch.zeros([batch_size_train,10])
            for i in range(batch_size_train):
                y[i,train_targets[i]] = 1

            loss = torch.mean( torch.square(y - z) )
            loss.backward()
            print(loss)

            optim.step()
            optim.zero_grad()
            # Model will not be saved
            torch.save(model,'MNISTModel6.pt')
            # print(train_data),print(train_targets)
            # exit(-1)



if __name__ == '__main__':
    train()