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

def evaluate(test_loader, model1, model2):
    model1 = cnn1
    model2 = cnn2
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

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        #images = Variable(images).cuda()
        outputs2 = model2(images)
        #outputs2 = F.softmax(logits2, dim=1)
        pred2 = (outputs2.data>.5).squeeze(1)
        total2 += labels.size(0)
        correct2 += (pred2 == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2

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

def co_teaching_loss(out1, out2, y, rt, criterion, pos_weight):
    #noreduce_loss = nn.BCELoss(reduction = 'none')
    noreduce_loss = bce_cmm(reduction = "none", pos_wt = pos_weight)
    loss_1 = noreduce_loss(out1, y.unsqueeze(1))
    loss_2 = noreduce_loss(out2, y.unsqueeze(1))
    
    #k = int(int(np.shape(loss_1)[0]) * rt)
    #_, model1_sm_idx = torch.topk(torch.tensor(loss_1.detach().numpy().transpose()), k=k, largest=False)
    #_, model2_sm_idx = torch.topk(torch.tensor(loss_2.detach().numpy().transpose()), k=k, largest=False)
    
    # add all indices of 1 to the mask
    #mask1 = torch.zeros_like(y, dtype = torch.bool)#np.zeros(np.shape(y.numpy())[0])
    #mask1[model1_sm_idx.squeeze(0)] = 1
    #mask1[y==1]=1

    #mask2 = torch.zeros_like(y, dtype = torch.bool)#np.zeros(np.shape(y.numpy())[0])
    #mask2[model2_sm_idx.squeeze(0)] = 1
    #mask2[y==1]=1

    k0 = int(int(sum(y==0))*rt)
    k1 = int(int(sum(y==1))*rt)
    _, model1_sm_idx1 = torch.topk(torch.tensor(loss_1.detach().numpy().transpose())*y, k=k1, largest=False)
    _, model1_sm_idx0 = torch.topk(torch.tensor(loss_1.detach().numpy().transpose())*(y-1)*-1, k=k0, largest=False)
    _, model2_sm_idx1 = torch.topk(torch.tensor(loss_2.detach().numpy().transpose())*y, k=k1, largest=False)
    _, model2_sm_idx0 = torch.topk(torch.tensor(loss_2.detach().numpy().transpose())*(y-1)*-1, k=k0, largest=False)

    mask1 = torch.zeros_like(y, dtype = torch.bool)
    mask2 = torch.zeros_like(y, dtype = torch.bool)
    mask1[model1_sm_idx1.squeeze(0)] = True
    mask1[model1_sm_idx0.squeeze(0)] = True
    mask2[model2_sm_idx1.squeeze(0)] = True
    mask2[model2_sm_idx0.squeeze(0)] = True

    out_1_sorted = out1[mask2]
    out_2_sorted = out2[mask1]
    y_1 = y[mask2]
    y_2 = y[mask1]

    # we passed in loss for each of the same data points from models 1 and 2, as vectors
    # for model 2 loss, we take only the smallest loss points from model 1, and vice versa
    # then we use these losses to update the models as normal
    loss_1_update = criterion(out_1_sorted.squeeze(1), y_1)
    loss_2_update = criterion(out_2_sorted.squeeze(1), y_2)
    
    return loss_1_update, loss_2_update


np.random.seed(229)
torch.manual_seed(229)

input_channel=1
num_classes=1
top_bn = False
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

coteach1 = torch.load("data/nn_mnist_coteach1_50epoch.pt")
coteach2 = torch.load("data/nn_mnist_coteach2_50epoch.pt")
baseline = torch.load("data/nn_mnist_baseline_50epoch.pt")
test_acc_baseline = np.loadtxt("data/test_acc_mnist_baseline_50epoch.csv")
test_acc_1 = np.loadtxt("data/test_acc_mnist_coteach1_50epoch.csv")
test_acc_2 = np.loadtxt("data/test_acc_mnist_coteach2_50epoch.csv")

train_loss_baseline = np.loadtxt("data/train_loss_mnist_baseline_50epoch.csv")
train_loss_coteach = np.loadtxt("data/train_loss_mnist_coteach_50epoch.csv")

# Make plots
plt.plot(np.transpose([test_acc_baseline,test_acc_1, test_acc_2]), 
                      label = ["Baseline", "Co-teaching 1", "Co-teaching 2"])
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
leg = plt.legend()
plt.show()

plt.plot(np.transpose([train_loss_baseline, train_loss_coteach]), 
                      label = ["Baseline", "Co-teaching"])
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
leg = plt.legend()
plt.show()

# evaluate model on test dataset
correct1 = 0
total1 = 0
coteach1.eval()
coteach2.eval()
for images, labels, _ in test_loader:
    #images = Variable(images).cuda()
    outputs1 = coteach1(images)
    outputs2 = coteach2(images)
    outputs = (np.transpose(outputs1.detach().numpy()).squeeze(0)+
                       np.transpose(outputs2.detach().numpy()).squeeze(0))/2
    #outputs1 = F.softmax(logits1, dim=1)
    pred1 = (outputs>.5)
    total1 += labels.size(0)
    correct1 += (pred1 == labels.numpy()).sum()
total1/correct1

correct1 = 0
total1 = 0
baseline.eval()
for images, labels, _ in test_loader:
    #images = Variable(images).cuda()
    outputs1 = baseline(images)
    outputs = np.transpose(outputs1.detach().numpy())
    #outputs1 = F.softmax(logits1, dim=1)
    pred1 = (outputs>.5)
    total1 += labels.size(0)
    correct1 += (pred1 == labels.numpy()).sum()

