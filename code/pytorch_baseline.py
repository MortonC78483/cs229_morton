# conda install numpy scipy pandas tensorflow
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
import wandb

sys.path.append('code')

def make_permuted_training(X_train, y_train, perc):
    # want to permute 10% of positive inspections to be 0 for facilities above EJSCREEN threshold
    positive_training = X_train[y_train == 1]
    num_to_permute = int(np.sum(positive_training["EJSCREEN_FLAG_US_Y"]>0)*perc)

    permute_indices = np.random.choice(np.where(np.logical_and(X_train["EJSCREEN_FLAG_US_Y"] >0, \
                                                               y_train == 1))[0],size = num_to_permute, \
                                                               replace = False)

    # reassign
    y_train_perm = copy.deepcopy(y_train)
    y_train_perm.iloc[permute_indices] = 0
    permuted_indicator = np.zeros(np.shape(y_train_perm)[0])
    permuted_indicator[permute_indices] = 1
    return y_train_perm, permuted_indicator

def make_train_test(path, batch_size):
    # load in data
    random_full = pandas.read_csv('data/random_full.csv')

    # split into x and y
    x_names = ["FAC_STATE", "FAC_CHESAPEAKE_BAY_FLG","FAC_INDIAN_CNTRY_FLG", "FAC_US_MEX_BORDER_FLG",\
                "FAC_FEDERAL_FLG", "FAC_PERCENT_MINORITY", "FAC_POP_DEN","AIR_FLAG", "SDWIS_FLAG", "RCRA_FLAG",\
                        "TRI_FLAG", "GHG_FLAG", "FAC_IMP_WATER_FLG","EJSCREEN_FLAG_US", "multiple_IDs","n_IDs",\
                                "SIC_AG_f", "SIC_MINE_f", "SIC_CONS_f", "SIC_MANU_f", "SIC_UTIL_f",\
                                    "SIC_WHOL_f", "SIC_RETA_f", "SIC_FINA_f", "SIC_SERV_f", "SIC_PUBL_f",\
                                        "num.facilities.cty", "num.facilities.st", "Party", "PERMIT_MAJOR",\
                                        "PERMIT_MINOR", "time.since.insp","prox.1yr", "prox.2yr", "prox.5yr"]
    random_full_x = pandas.get_dummies(random_full[x_names], drop_first=True)
    random_full_y = random_full["DV"]

    # make train/test split
    X_train, X_test, y_train, y_test = train_test_split(random_full_x,random_full_y,\
                                                        test_size=0.20,random_state=229)

    y_train_10, perm_indic = make_permuted_training(X_train, y_train, .3)
    colnames = X_train.columns.values
    # fit scaler on training data
    scaler = StandardScaler() 
    scaler.fit(X_train)  
    X_train[colnames] = scaler.transform(X_train[colnames])  
    # apply same transformation to test data
    X_test[colnames] = scaler.transform(X_test[colnames]) 
    X_train = pandas.DataFrame(X_train, columns = colnames)
    X_test = pandas.DataFrame(X_test, columns = colnames)
    X_train.index = y_train_10.index
    X_test.index = y_test.index

    # reorder training data
    #train_perm = torch.randperm(X_train.size()[0])
    X_train = torch.tensor(X_train.values)
    X_train = X_train.to(torch.float32)
    #X_train = X_train[train_perm]
    y_train = torch.tensor(y_train_10.values)
    y_train = y_train.type(torch.LongTensor)
    #y_train = y_train[train_perm]
    train_data = torch.utils.data.TensorDataset(X_train, y_train)

    X_test = torch.tensor(X_test.values)
    X_test = X_test.to(torch.float32)
    y_test = torch.tensor(y_test.values)
    y_test = y_test.type(torch.LongTensor)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return[trainloader, testloader,X_train, y_train, X_test, y_test, perm_indic]

########################################################
################### PYTORCH ############################
########################################################

# https://discuss.pytorch.org/t/indices-of-a-dataset-sampled-by-dataloader/47748/2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(86, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256) # delete for 2 layer
        self.fc4 = nn.Linear(256, 256) # delete for 2 layer
        self.fc5 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # delete for 2 layer
        x = F.relu(self.fc4(x)) # delete for 2 layer
        x = self.fc5(x)
        return torch.sigmoid(x)

if __name__ == '__main__':
    learning_rate = .001
    epochs = 20
    batch_size = 256

    #wandb.init(project="cs229-project", config={"learning_rate": learning_rate, "epochs": epochs})
    
    random.seed(229)
    np.random.seed(229)
    torch.manual_seed(229)

    [trainloader, testloader,X_train, y_train, X_test, y_test, perm_indic] = make_train_test('data/random_full.csv', batch_size)

    net = Net()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_list = []
    oneloss_list = []
    train_vec = []
    test_vec = []

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # train loss
            #train_loss = criterion(net(X_train).squeeze(1), y_train.float())
            #train_vec.append(train_loss.item())
            # test loss
            #test_loss = criterion(net(X_test).squeeze(1), y_test.float())
            #test_vec.append(test_loss.item())
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                oneloss_list.append(running_loss)
                running_loss = 0.0
                
        # calculate confidence on all training data where true assignment is 0
        true_negatives =  net(X_train[(y_train.numpy() == 0) & (perm_indic == 0)])
        loss_true_negatives = criterion(true_negatives, y_train[(y_train.numpy() == 0) & (perm_indic == 0)].unsqueeze(1).float()).detach().item()
        false_negatives =  net(X_train[(y_train.numpy() == 0) & (perm_indic == 1)])
        loss_false_negatives = criterion(false_negatives, y_train[(y_train.numpy() == 0) & (perm_indic == 1)].unsqueeze(1).float()).detach().item()
        loss_list.append([loss_true_negatives, loss_false_negatives])


    plt.plot(loss_list, label = ["True negatives", "False negatives"])
    plt.ylabel("Loss at end of each epoch (model 1)")
    plt.xlabel("Epoch")
    leg = plt.legend()
    plt.show()

    torch.save(net, "data/nn_20epoch_4layer_256_30pct2.pt")
    loss_list
        