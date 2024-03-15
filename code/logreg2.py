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
from sklearn.linear_model import LogisticRegression

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
    X_test_orig = pandas.DataFrame.copy(X_test)

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
    return[X_test_orig, trainloader, testloader,X_train, y_train, X_test, y_test, perm_indic]



def main():
    # Hyperparameters
    
    batch_size = 256
    random.seed(229)
    np.random.seed(229)
    torch.manual_seed(229)
    perc = 0

    [X_test_orig, trainloader, testloader,X_train, y_train, X_test, y_test, perm_indic] = make_train_test('data/random_full.csv', batch_size, perc)

    logreg = LogisticRegression(solver='liblinear', random_state=229, max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict_proba(X_test)[:,1]

    #n_true_pos = sum(y_test).item()
    n_true_pos = int(len(y_test)/10)
    # get highest values in mean_pred
    _,indices = torch.topk(torch.tensor(logreg_pred), k=n_true_pos, largest=True)
    sum(y_test[indices])/n_true_pos

    # proportion of recommended inspections in EJ areas
    X_test_orig = X_test_orig.reset_index()
    sum(X_test_orig.loc[indices.numpy(), "EJSCREEN_FLAG_US_Y"]>0)/len(y_test[indices.numpy()])

    # proportion of true positives in EJ areas
    sum(y_test[np.intersect1d(np.where(X_test_orig["EJSCREEN_FLAG_US_Y"]>0)[0], indices.numpy())])
