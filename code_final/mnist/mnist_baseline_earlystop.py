import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas
import random
import sys
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import torchvision.transforms as transforms
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

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def val_step(valloader, model, optimizer, criterion):
    global_step = 0
    loss = 0.0

    model = model.train()
    for i, data in enumerate(valloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _ = data

        outputs = model(inputs)
        labels = labels.unsqueeze(1)
        model_loss = criterion(outputs, labels.float())

        loss += (model_loss.item())
        global_step += 1
    
    return loss/global_step

np.random.seed(229)
torch.manual_seed(229)

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
# make validation
val_len = int(len(train_dataset.train_labels)/10)
val_dataset = train_dataset
val_dataset.val_labels = val_dataset.train_labels[:val_len]
train_dataset.train_labels = train_dataset.train_labels[val_len:]

val_dataset.val_data = val_dataset.train_data[:val_len]
train_dataset.train_data = train_dataset.train_data[val_len:]

test_dataset = MNIST(root='./data/',
                    download=False,  
                    train=False, 
                    transform=transforms.ToTensor()
                        )

test_dataset.test_data = test_dataset.test_data[(test_dataset.test_labels == 0) + (test_dataset.test_labels == 1)]
test_dataset.test_labels = test_dataset.test_labels[(test_dataset.test_labels == 0) + (test_dataset.test_labels == 1)]

input_channel=1
num_classes=1
top_bn = False
epoch_decay_start = 80
n_epoch = 200
batch_size = 256
learning_rate = .001
pos_weight = 1

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               drop_last=True,
                                               shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
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
early_stopper = EarlyStopper(patience=3, min_delta=0)
val_losses = []
for epoch in range(29, n_epoch):
    # train models
    cnn1.train()
    avg_loss, cnn1=train_step(train_loader, cnn1, optimizer1, criterion, 1)
    # validation
    val_loss = val_step(val_loader, cnn1, optimizer1, criterion)
    #if early_stopper.early_stop(val_loss):             
    #    break
    val_losses.append(val_loss)
    train_loss.append(avg_loss)
    # evaluate models
    test_acc1=evaluate(test_loader, cnn1)
    test_acc.append(test_acc1)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %%' % (epoch+1, n_epoch, len(test_dataset), test_acc1))

np.savetxt("data/val_losses_mnist_baseline.csv",val_losses)