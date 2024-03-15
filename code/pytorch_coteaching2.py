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

    y_train_10, perm_indic = make_permuted_training(X_train, y_train, perc)
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

def train_step(trainloader, model_list, optimizer1, optimizer2, criterion, rt, pos_wt):
    global_step = 0
    avg_loss = 0.0

    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()

    for x, y in trainloader:
        x, y = x.float(), y.float()

        out1 = model1(x)
        out2 = model2(x)

        model1_loss, model2_loss = co_teaching_loss(out1, out2, y, rt, criterion, pos_wt)

        optimizer1.zero_grad()
        model1_loss.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        model2_loss.backward()
        optimizer2.step()

        avg_loss += (model1_loss.item() + model2_loss.item())/2
        global_step += 1

    return avg_loss / global_step, [model1, model2]

def test_model(model1, model2, testloader, criterion):
    model1.eval()
    model2.eval()
    test_loss = 0
    output_vec = torch.empty((0), dtype=torch.float32)
    y_vec = torch.empty((0), dtype=torch.float32)
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.float(), y.float()
            outputs1 = model1(x)
            outputs2 = model2(x)
            #outputs = (outputs1+outputs2)/2
            outputs = torch.max(outputs1, outputs2)
            output_vec = torch.cat((output_vec, outputs), 0)
            y_vec = torch.cat((y_vec, y), 0)
            test_loss += criterion(outputs, y.unsqueeze(1)).item()
            #predicted = torch.round(outputs)
            #total += y.size(0)
            #correct += (predicted == y.unsqueeze(1)).sum().item()
    n_true_pos = int(len(y_vec)/10)
    _,indices = torch.topk(output_vec.squeeze(1), k=n_true_pos, largest=True)
    accuracy = sum(y_vec.numpy()[indices])/len(y_vec.numpy()[indices])
    test_loss /= len(testloader)
    #accuracy = correct / total
    return test_loss, accuracy

def main():
    # Hyperparameters
    learning_rate = 0.001
    epochs = 20
    batch_size = 256
    pos_weight = 1
    #perc = 0

    percs = [0, .3, .5]
    for perc in percs:
        random.seed(229)
        np.random.seed(229)
        torch.manual_seed(229)

        [trainloader, testloader,X_train, y_train, X_test, y_test, perm_indic] = make_train_test('data/random_full.csv', batch_size, perc)

        # Initialize models, optimizer, and criterion
        model1 = Net()
        model2 = Net()
        #optimizer1 = optim.Adam(list(model1.parameters()), lr=learning_rate)
        #optimizer2 = optim.Adam(list(model2.parameters()), lr=learning_rate)
        optimizer1 = optim.SGD(list(model1.parameters()), lr=learning_rate, momentum = .9)
        optimizer2 = optim.SGD(list(model2.parameters()), lr=learning_rate, momentum = .9)
        #criterion = nn.BCELoss()
        criterion = bce_cmm(pos_wt = pos_weight)

        loss_list = []
        accuracy_list = []
        for epoch in range(epochs):
            if epoch < 10:
                rt = 1
            else:
                rt = 1-.3*min(epoch/epochs, 1)
            avg_loss, [model1, model2] = train_step(trainloader, [model1, model2], optimizer1, optimizer2, criterion, rt, pos_weight)
            # calculate confidence on all training data where true assignment is 0
            #true_negatives =  (model1(X_train[(y_train.numpy() == 0) & (perm_indic == 0)]) + model2(X_train[(y_train.numpy() == 0) & (perm_indic == 0)]))/2
            true_negatives = torch.max(model1(X_train[(y_train.numpy() == 0) & (perm_indic == 0)]), model2(X_train[(y_train.numpy() == 0) & (perm_indic == 0)]))
            loss_true_negatives = criterion(true_negatives, y_train[(y_train.numpy() == 0) & (perm_indic == 0)].unsqueeze(1).float()).detach().item()
            #false_negatives =  (model1(X_train[(y_train.numpy() == 0) & (perm_indic == 1)]) + model2(X_train[(y_train.numpy() == 0) & (perm_indic == 1)]))/2
            false_negatives = torch.max(model1(X_train[(y_train.numpy() == 0) & (perm_indic == 1)]), model2(X_train[(y_train.numpy() == 0) & (perm_indic == 1)]))
            loss_false_negatives = criterion(false_negatives, y_train[(y_train.numpy() == 0) & (perm_indic == 1)].unsqueeze(1).float()).detach().item()
            #positives = (model1(X_train[(y_train.numpy() == 1)]) + model2(X_train[(y_train.numpy() == 1)]))/2
            #loss_positives = criterion(positives, y_train[(y_train.numpy() == 1)].unsqueeze(1).float()).detach().item()
            loss_list.append([loss_true_negatives, loss_false_negatives])#, loss_positives])
            
            test_loss, accuracy = test_model(model1, model2, testloader, criterion)
            accuracy_list.append([accuracy])

            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}, Test Loss: {test_loss}, Test Accuracy: {accuracy}")

        plt.plot(loss_list, label = ["True negatives", "False negatives"])#, "Positives"])
        plt.ylabel("Loss at end of each epoch")
        plt.xlabel("Epoch")
        leg = plt.legend()
        plt.show()

        plt.plot(accuracy_list)
        plt.ylabel("TP/(TP+FP)")
        plt.xlabel("Epoch")
        plt.show()

        #torch.save(model1, "data/nn_coteaching1_"+str(perc)+"pct.pt")
        #torch.save(model2, "data/nn_coteaching2_"+str(perc)+"pct.pt")
        np.savetxt("data/nn_coteaching1_"+str(perc)+"pct_accuracy.csv", accuracy_list)
        np.savetxt("data/nn_coteaching1_"+str(perc)+"pct_loss.csv", loss_list)
        loss_list

if __name__ == "__main__":
    main()

