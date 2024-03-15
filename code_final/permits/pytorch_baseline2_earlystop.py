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

def make_train_test(path, batch_size, perc):
    # load in data
    random_full = pandas.read_csv(path)

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
    
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,\
                                                        test_size=0.10,random_state=229)
    
    y_train_10, perm_indic_train = make_permuted_training(X_train, y_train, perc)
    y_val_10, perm_indic_val = make_permuted_training(X_val, y_val, perc)
    colnames = X_train.columns.values
   
    # fit scaler on training data
    scaler = StandardScaler() 
    scaler.fit(X_train)  
    X_train[colnames] = scaler.transform(X_train[colnames])  
    # apply same transformation to test data
    X_test[colnames] = scaler.transform(X_test[colnames]) 
    X_val[colnames] = scaler.transform(X_val[colnames])
    X_train = pandas.DataFrame(X_train, columns = colnames)
    X_test = pandas.DataFrame(X_test, columns = colnames)
    X_val = pandas.DataFrame(X_val, columns = colnames)
    X_train.index = y_train_10.index
    X_test.index = y_test.index
    X_val.index = y_val_10.index

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

    X_val = torch.tensor(X_val.values)
    X_val = X_val.to(torch.float32)
    y_val = torch.tensor(y_val_10.values)
    y_val = y_val.type(torch.LongTensor)
    val_data = torch.utils.data.TensorDataset(X_val, y_val)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return[trainloader, testloader,valloader, X_train, y_train, X_test, y_test, X_val, y_val,perm_indic_train, perm_indic_val]

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

def train_step(trainloader, model, optimizer, criterion):
    global_step = 0
    loss = 0.0

    model = model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        labels = labels.unsqueeze(1)
        model_loss = criterion(outputs, labels.float())
        model_loss.backward()
        optimizer.step()

        loss += (model_loss.item())
        global_step += 1
    return loss/global_step, model

def val_step(valloader, model, optimizer, criterion):
    global_step = 0
    loss = 0.0

    model = model.train()
    for i, data in enumerate(valloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        outputs = model(inputs)
        labels = labels.unsqueeze(1)
        model_loss = criterion(outputs, labels.float())

        loss += (model_loss.item())
        global_step += 1
    return loss/global_step

def test_model(model, testloader, criterion):
    model.eval()
    test_loss = 0
    output_vec = torch.empty((0), dtype=torch.float32)
    y_vec = torch.empty((0), dtype=torch.float32)
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.float(), y.float()
            outputs = model(x)
            output_vec = torch.cat((output_vec, outputs), 0)
            y_vec = torch.cat((y_vec, y), 0)
            test_loss += criterion(outputs, y.unsqueeze(1)).item()
    n_true_pos = int(len(y_vec)/10)
    _,indices = torch.topk(output_vec.squeeze(1), k=n_true_pos, largest=True)
    accuracy = sum(y_vec.numpy()[indices])/len(y_vec.numpy()[indices])
    test_loss /= len(testloader)
    return test_loss, accuracy

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

def main():
    learning_rate = .001
    epochs = 20
    batch_size = 256
    pos_weight = 1
    perc = 0
    #wandb.init(project="cs229-project", config={"learning_rate": learning_rate, "epochs": epochs})
    
    random.seed(229)
    np.random.seed(229)
    torch.manual_seed(229)

    [trainloader, testloader,valloader,X_train, y_train, X_test, y_test, X_val, y_val,perm_indic, perm_indic_val] = make_train_test('data/random_full.csv', batch_size,perc)
    early_stopper = EarlyStopper(patience=3, min_delta=0)
    net = Net()
    #criterion = nn.BCELoss()
    criterion = bce_cmm(pos_wt = pos_weight)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_list = []
    accuracy_list = []
    val_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss, net = train_step(trainloader, net, optimizer, criterion)
        
        # validation
        val_loss = val_step(valloader, net, optimizer, criterion)
        if early_stopper.early_stop(val_loss):             
            break
        # calculate confidence on all training data where true assignment is 0
        true_negatives =  net(X_train[(y_train.numpy() == 0) & (perm_indic == 0)])
        loss_true_negatives = criterion(true_negatives, y_train[(y_train.numpy() == 0) & (perm_indic == 0)].unsqueeze(1).float()).detach().item()
        false_negatives =  net(X_train[(y_train.numpy() == 0) & (perm_indic == 1)])
        loss_false_negatives = criterion(false_negatives, y_train[(y_train.numpy() == 0) & (perm_indic == 1)].unsqueeze(1).float()).detach().item()
        val_list.append(val_loss)
        #positives = net(X_train[(y_train.numpy() == 1)])
        #loss_positives = criterion(positives, y_train[(y_train.numpy() == 1)].unsqueeze(1).float()).detach().item()
        loss_list.append([loss_true_negatives, loss_false_negatives])#, loss_positives])
        
        test_loss, accuracy = test_model(net, testloader, criterion)
        accuracy_list.append([accuracy])

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {accuracy}")

    plt.plot(loss_list, label = ["True negatives", "False negatives"])#, "Positives"])
    plt.ylabel("Loss at end of each epoch (model 1)")
    plt.xlabel("Epoch")
    leg = plt.legend()
    plt.show()

    plt.plot(accuracy_list)
    plt.ylabel("TP/(TP+FP)")
    plt.xlabel("Epoch")
    plt.show()

    torch.save(net, "data/nn_baseline_"+str(perc)+"pct.pt")
    loss_list

if __name__ == '__main__':
    main()