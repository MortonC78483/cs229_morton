import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import sys
import numpy as np
sys.path.append('code')
from mnist import MNIST

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=1, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        #self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        #self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        #self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        #self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        #self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        #self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        #self.bn4=nn.BatchNorm2d(256)
        #self.bn5=nn.BatchNorm2d(256)
        #self.bn6=nn.BatchNorm2d(256)
        #self.bn7=nn.BatchNorm2d(512)
        #self.bn8=nn.BatchNorm2d(256)
        #self.bn9=nn.BatchNorm2d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        #h=self.c4(h)
        #h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        #h=self.c5(h)
        #h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        #h=self.c6(h)
        #h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        #h=F.max_pool2d(h, kernel_size=2, stride=2)
        #h=F.dropout2d(h, p=self.dropout_rate)

        #h=self.c7(h)
        #h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        #h=self.c8(h)
        #h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        #h=self.c9(h)
        #h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return torch.sigmoid(logit)

def evaluate(test_loader, model1):
    model1 = cnn1
    print ("Evaluating...")
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        #images = Variable(images).cuda()
        outputs1 = model1(images)
        #outputs1 = F.softmax(logits1, dim=1)
        pred1 = (outputs1.data>.5).squeeze(1)
        total1 += labels.size(0)
        correct1 += (pred1 == labels).sum()

    acc1 = 100*float(correct1)/float(total1)
    return acc1

class bce_cmm():
    def __init__(self, reduction = "mean", pos_wt=1):
        super(bce_cmm, self).__init__()
        self.reduction = reduction
        self.pos_wt = pos_wt
    def __call__(self, input, target):
        loss = self.binary_cross_entropy(input, target)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Supported modes are 'none', 'mean', and 'sum'.")
    def binary_cross_entropy(self, input, target):
        #eps=np.exp(-100)
        #input_scaled = F.relu(input)
        #loss = -target * self.pos_wt * torch.clamp(torch.log(input + eps), min = -100) - (1 - target) * torch.clamp(torch.log(1 - input + eps), min = -100)
        weighted_target = torch.mul(target,self.pos_wt)
        bceloss = nn.BCELoss(reduction = 'none')
        loss_pos = bceloss(input, target)* weighted_target
        loss_neg = bceloss(input, target)*torch.mul(torch.subtract(target,1), -1)
        loss = torch.add(loss_pos, loss_neg)
        return loss

def fn_loss(out1, y, criterion):
    #noreduce_loss = nn.BCELoss(reduction = 'none')
    #noreduce_loss = bce_cmm(reduction = "none", pos_wt = pos_weight)
    #loss_1 = noreduce_loss(out1, y.unsqueeze(1))

    # we passed in loss for each of the same data points from models 1 and 2, as vectors
    # for model 2 loss, we take only the smallest loss points from model 1, and vice versa
    # then we use these losses to update the models as normal
    loss_1_update = criterion(out1.squeeze(1), y)
    
    return loss_1_update

def train_step(trainloader, model1, optimizer1, criterion, pos_wt):
    global_step = 0
    avg_loss = 0.0
    #model_list = [cnn1, cnn2]
    model1 = model1.train()
    #trainloader = train_loader
    for i, (x,y, indexes) in enumerate(trainloader):
        #print(i)
        x, y = x.float(), y.float()

        out1 = model1(x)

        model1_loss = fn_loss(out1, y, criterion)

        optimizer1.zero_grad()
        model1_loss.backward()
        optimizer1.step()

        avg_loss += (model1_loss.item())
        global_step += 1

    return avg_loss / global_step, model1


np.random.seed(229)
torch.manual_seed(229)

input_channel=1
num_classes=1
top_bn = False
epoch_decay_start = 80
n_epoch = 200
batch_size = 256
learning_rate = .001
pos_weight = 1
# have already downloaded
train_dataset = MNIST(root='./data/',
                    download=False,  
                    train=True, 
                    transform=transforms.ToTensor()
                )
train_dataset.train_data = train_dataset.train_data[(train_dataset.train_labels == 0) + (train_dataset.train_labels == 1)]
train_dataset.train_labels = train_dataset.train_labels[(train_dataset.train_labels == 0) + (train_dataset.train_labels == 1)]

# change around training labels
n_change_0 = int(sum(train_dataset.train_labels == 0)*.45)
n_change_1 = int(sum(train_dataset.train_labels == 1)*.45)
sum_0 = 0
sum_1 = 0
for i in range(0, len(train_dataset.train_labels)):
    if (sum_0 < n_change_0) and  train_dataset.train_labels[i] == 0:
        train_dataset.train_labels[i] = 1
        sum_0 += 1
    elif (sum_1 < n_change_1) and train_dataset.train_labels[i] == 1:
        train_dataset.train_labels[i] = 0
        sum_1 += 1

test_dataset = MNIST(root='./data/',
                    download=False,  
                    train=False, 
                    transform=transforms.ToTensor()
                        )

test_dataset.test_data = test_dataset.test_data[(test_dataset.test_labels == 0) + (test_dataset.test_labels == 1)]
test_dataset.test_labels = test_dataset.test_labels[(test_dataset.test_labels == 0) + (test_dataset.test_labels == 1)]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               drop_last=True,
                                               shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              drop_last=True,
                                              shuffle=False)

cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)

criterion = bce_cmm(pos_wt = pos_weight)

n_epoch = 50
train_loss = []
test_acc = []
for epoch in range(0, n_epoch):
    # train models
    cnn1.train()
    avg_loss, cnn1=train_step(train_loader, cnn1, optimizer1, criterion, 1)
    train_loss.append(avg_loss)
    # evaluate models
    test_acc1=evaluate(test_loader, cnn1)
    test_acc.append(test_acc1)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %%' % (epoch+1, n_epoch, len(test_dataset), test_acc1))

torch.save(cnn1, "data/nn_mnist_baseline_50epoch.pt")
np.savetxt("data/train_loss_mnist_baseline_50epoch.csv", train_loss)
np.savetxt("data/test_acc_mnist_baseline_50epoch.csv", test_acc)
