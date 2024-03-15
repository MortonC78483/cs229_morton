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

# for an already fit model
def summarize_model(X_train, y_train, X_test, y_test, X_test_orig, model, name, save = True):

    predicted_probs = model(X_train).detach().numpy().transpose().squeeze(0)

    # get optimal threshold
    preds = predicted_probs
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_train.numpy(), preds)
    #roc_auc = sklearn.metrics.auc(fpr, tpr)
    threshold_opt = threshold[np.argmax(tpr - fpr)] 

    # save ROC curve with optimal threshold point
    if save:
        plt.clf()
        plt.plot(fpr, tpr)
        plt.plot([fpr[np.argmax(tpr - fpr)]],[tpr[np.argmax(tpr - fpr)]],'o')
        plt.savefig("outputs/" + "model1" + '.png')

    # run on test data
    test_probs = model(X_test).detach().numpy().transpose().squeeze(0)
    test_pred = (test_probs >= threshold_opt).astype(bool)
    sklearn.metrics.confusion_matrix(y_test, test_pred)
    #print(sklearn.metrics.classification_report(y_test, test_pred))

    # accuracy overall
    overall_accuracy = (1-(y_test - test_pred) ** 2).sum()/len(y_test)

    # check proportion of test facilities it gets right in and outside of EJSCREEN areas
    test_ejscreen_indices = np.where(X_test_orig["EJSCREEN_FLAG_US_Y"] >0)
    y_test_ejscreen = y_test.numpy()[test_ejscreen_indices]
    test_pred_ejscreen = test_pred[test_ejscreen_indices]
    ejscreen_accuracy = (1-(y_test_ejscreen - test_pred_ejscreen) ** 2).sum()/len(y_test_ejscreen)

    test_noejscreen_indices = np.where(X_test_orig["EJSCREEN_FLAG_US_Y"] <= 0)
    y_test_noejscreen = y_test.numpy()[test_noejscreen_indices]
    test_pred_noejscreen = test_pred[test_noejscreen_indices]
    noejscreen_accuracy = (1-(y_test_noejscreen - test_pred_noejscreen) ** 2).sum()/len(y_test_noejscreen)
    
    # calculate accuracy on positive areas (TP/(FN+TP))
    #overall_accuracy_pos = sum(np.logical_and(y_test == 1, test_pred == 1))/sum(y_test)
    ej_accuracy_pos = sum(np.logical_and(y_test.numpy() == 1, \
                                         np.logical_and(test_pred == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] >0)))\
                                            /sum(np.logical_and(y_test.numpy() == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] >0))
    noej_accuracy_pos = sum(np.logical_and(test_pred == 1, np.logical_and(y_test.numpy() == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] <= 0)))\
                                            /sum(np.logical_and(y_test.numpy() == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] <= 0))
    ej_accuracy_neg = sum(np.logical_and(y_test.numpy() == 0, \
                                         np.logical_and(test_pred == 0, X_test_orig["EJSCREEN_FLAG_US_Y"] >0)))\
                                            /sum(np.logical_and(y_test.numpy() == 0, X_test_orig["EJSCREEN_FLAG_US_Y"] >0))
    noej_accuracy_neg = sum(np.logical_and(y_test.numpy() == 0, \
                                           np.logical_and(test_pred == 0, X_test_orig["EJSCREEN_FLAG_US_Y"] <= 0)))\
                                            /sum(np.logical_and(y_test.numpy() == 0, X_test_orig["EJSCREEN_FLAG_US_Y"] <= 0))
    
    #acc = np.array(["overall_accuracy", overall_accuracy])
    #ej_acc = np.array(["ejscreen_accuracy", ejscreen_accuracy])
    #noej_acc = np.array(["noejscreen_accuracy", noejscreen_accuracy])
    #acc_pos = np.array(["overall accuracy, positive", overall_accuracy_pos])
    #ej_acc_pos = np.array(["overall accuracy, ej positive", ej_accuracy_pos])
    #noej_acc_pos = np.array(["overall accuracy, no ej positive", noej_accuracy_pos])
    #pred_pos = np.array(["predicted/true positives", sum(test_pred)/sum(y_test)])
    #pred_pos_ej = np.array(["predicted/true positives, ejscreen", np.mean(test_pred_ejscreen)\
    #                        /np.mean(y_test_ejscreen)])
    #pred_pos_noej = np.array(["predicted/true positives, no ejscreen",np.mean(test_pred_noejscreen)\
    #                          /np.mean(y_test_noejscreen)])
    #ratio_ej_noej = np.array(["ratio of ej to non-ej predicted positives",np.mean(test_pred_ejscreen)\
    #                          /np.mean(test_pred_noejscreen)])
    #true_ratio_ej_noej = np.array(["ratio of true ej to non-ej positives",np.mean(y_test_ejscreen)\
    #                          /np.mean(y_test_noejscreen)])
    balanced_acc_ej = np.array(["balanced accuracy, ej", 1/2*(ej_accuracy_pos + ej_accuracy_neg)])
    balanced_acc_noej = np.array(["balanced accuracy, no ej", 1/2*(noej_accuracy_pos + noej_accuracy_neg)])
    pred_pos_ej_n = np.array(["n positive, ej", sum(np.logical_and(test_pred == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] >0))])
    pred_pos_noej_n = np.array(["n positive, no ej", sum(np.logical_and(test_pred == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] <= 0))])
    #tosave = np.array([acc,ej_acc,noej_acc,acc_pos,\
    #                   ej_acc_pos,noej_acc_pos,pred_pos,\
    #                   pred_pos_ej,pred_pos_noej,ratio_ej_noej,true_ratio_ej_noej])
    tosave = np.array([balanced_acc_ej, balanced_acc_noej, pred_pos_ej_n, pred_pos_noej_n])
    if save:
        np.savetxt("outputs/" + name + ".txt", tosave, fmt='%s')

    #return np.mean(test_pred_ejscreen)/np.mean(test_pred_noejscreen),\
        #np.mean(y_test_ejscreen)/np.mean(y_test_noejscreen)
    return sum(np.logical_and(test_pred == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] >0))/sum(np.logical_and(test_pred == 1, X_test_orig["EJSCREEN_FLAG_US_Y"] <= 0))

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

def eval_coteaching(X_test_orig, X_test, y_test, path1, path2):
    model1 = torch.load(path1)
    model2 = torch.load(path2)
    # get model 1 predictions
    model1_pred = model1(X_test)
    # get model 2 predictions
    model2_pred = model2(X_test)

    #mean_pred = (model1_pred.detach().numpy().squeeze(1)+model2_pred.detach().numpy().squeeze(1))/2
    mean_pred = torch.max(model1_pred.detach().squeeze(1),model2_pred.detach().squeeze(1)).numpy()
    # number of true positives
    #n_true_pos = sum(y_test).item()
    n_true_pos = int(len(y_test)/10)
    #n_true_pos= int(len(y_test.numpy())*.05)
    # get highest values in mean_pred
    _,indices = torch.topk(torch.tensor(mean_pred), k=n_true_pos, largest=True)
    #y_test = y_test.reset_index(inplace=True, drop=True)
    X_test_orig.reset_index(inplace=True, drop=True)
    val_1 = sum(y_test.numpy()[indices])/len(y_test.numpy()[indices])

    # proportion of recommended inspections in EJ areas
    # number of recommended inspections in EJ areas/number of recommended inspections
    val_2 = sum(X_test_orig.loc[indices.numpy(), "EJSCREEN_FLAG_US_Y"]>0)/len(y_test[indices.numpy()])

    # number of recommended inspections in EJ areas that are positive
    val_3 = sum(y_test[np.intersect1d(np.where(X_test_orig["EJSCREEN_FLAG_US_Y"]>0)[0], indices.numpy())])
    return [val_1, val_2, val_3]

def eval_baseline(X_test_orig, X_test, y_test, path):
    nocoteach = torch.load(path)  
    n_true_pos = int(len(y_test)/10)
    # get basic nn
    nocoteach_pred = nocoteach(X_test)

    nocoteach_pred = (nocoteach_pred.detach().numpy().squeeze(1))

    # get highest values in mean_pred
    _,indices = torch.topk(torch.tensor(nocoteach_pred), k=n_true_pos, largest=True)
    X_test_orig.reset_index(inplace=True, drop=True)
    val_4 = sum(y_test.numpy()[indices])/len(y_test.numpy()[indices])

    # proportion of recommended inspections in EJ areas
    val_5 = sum(X_test_orig.loc[indices.numpy(), "EJSCREEN_FLAG_US_Y"]>0)/len(y_test[indices.numpy()])

    # proportion of true positives in EJ areas
    val_6= sum(y_test[np.intersect1d(np.where(X_test_orig["EJSCREEN_FLAG_US_Y"]>0)[0], indices.numpy())])

    return [val_4, val_5, val_6]

########################################################
################### ANALYZE ############################
########################################################
def main():
    # Hyperparameters
    learning_rate = 0.001
    epochs = 20
    batch_size = 256

    random.seed(229)
    np.random.seed(229)
    torch.manual_seed(229)


    sys.path.append('code')

    [X_test_orig, trainloader, testloader,X_train, y_train, X_test, y_test, perm_indic] = make_train_test('data/random_full.csv', 256)

    
    eval_baseline(X_test_orig, X_test, y_test, "data/nn_baseline_0pct.pt")
    eval_coteaching(X_test_orig, X_test, y_test, "data/nn_coteaching1_0pct.pt", "data/nn_coteaching2_0pct.pt")

    eval_baseline(X_test_orig, X_test, y_test, "data/nn_baseline_0.3pct.pt")
    eval_coteaching(X_test_orig, X_test, y_test, "data/nn_coteaching1_0.3pct.pt", "data/nn_coteaching2_0.3pct.pt")

    eval_baseline(X_test_orig, X_test, y_test, "data/nn_baseline_0.5pct.pt")
    eval_coteaching(X_test_orig, X_test, y_test, "data/nn_coteaching1_0.5pct.pt", "data/nn_coteaching2_0.5pct.pt")

    path1 = "data/nn_coteaching1_0.3pct.pt"
    path2 = "data/nn_coteaching2_0.3pct.pt"
    model1 = torch.load(path1)
    model2 = torch.load(path2)
    # get model 1 predictions
    model1_pred = model1(X_test)
    # get model 2 predictions
    model2_pred = model2(X_test)

    #mean_pred = (model1_pred.detach().numpy().squeeze(1)+model2_pred.detach().numpy().squeeze(1))/2
    mean_pred = torch.max(model1_pred.detach().squeeze(1),model2_pred.detach().squeeze(1)).numpy()
    # number of true positives
    #n_true_pos = sum(y_test).item()
    n_true_pos = int(len(y_test)/10)
    n_EJ_pos = int(n_true_pos*.2)
    n_noEJ_pos = n_true_pos-n_EJ_pos

    mean_pred_EJ = mean_pred*np.array(X_test_orig["EJSCREEN_FLAG_US_Y"]>0)
    mean_pred_noEJ = mean_pred*np.array(X_test_orig["EJSCREEN_FLAG_US_Y"]<=0)

    _,indicesEJ = torch.topk(torch.tensor(mean_pred_EJ), k=n_EJ_pos, largest=True)
    _,indicesnoEJ = torch.topk(torch.tensor(mean_pred_noEJ), k=n_noEJ_pos, largest=True)

    indices = torch.cat((indicesEJ, indicesnoEJ))
    X_test_orig.reset_index(inplace=True, drop=True)
    val_1 = sum(y_test.numpy()[indices])/len(y_test.numpy()[indices])

    # proportion of recommended inspections in EJ areas
    # number of recommended inspections in EJ areas/number of recommended inspections
    val_2 = sum(X_test_orig.loc[indices.numpy(), "EJSCREEN_FLAG_US_Y"]>0)/len(y_test[indices.numpy()])

    # number of recommended inspections in EJ areas that are positive
    val_3 = sum(y_test[np.intersect1d(np.where(X_test_orig["EJSCREEN_FLAG_US_Y"]>0)[0], indices.numpy())])
    return [val_1, val_2, val_3]

    
    