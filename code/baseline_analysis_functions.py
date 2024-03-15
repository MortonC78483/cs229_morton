# functions to try to fix formatting
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas
import random
import copy

##############################
##### FUNCTION ##############################
##############################
def run_model(X_train, y_train, X_test, y_test, model, name, save = True):
    # create model
    model.fit(X_train, y_train)

    # overall accuracy
    model.score(X_train, y_train)

    predicted_probs = model.predict_proba(X_train)

    # get optimal threshold
    preds = predicted_probs[:,1]
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_train, preds)
    #roc_auc = sklearn.metrics.auc(fpr, tpr)
    threshold_opt = threshold[np.argmax(tpr - fpr)] 

    # save ROC curve with optimal threshold point
    if save:
        plt.clf()
        plt.plot(fpr, tpr)
        plt.plot([fpr[np.argmax(tpr - fpr)]],[tpr[np.argmax(tpr - fpr)]],'o')
        plt.savefig("outputs/" + name + '.png')

    # run on test data
    test_pred = (model.predict_proba(X_test)[:,1] >= threshold_opt).astype(bool)
    sklearn.metrics.confusion_matrix(y_test, test_pred)
    #print(sklearn.metrics.classification_report(y_test, test_pred))

    # accuracy overall
    overall_accuracy = (1-(y_test - test_pred) ** 2).sum()/len(y_test)

    # check proportion of test facilities it gets right in and outside of EJSCREEN areas
    test_ejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] >0)
    y_test_ejscreen = y_test.iloc[test_ejscreen_indices]
    test_pred_ejscreen = test_pred[test_ejscreen_indices]
    ejscreen_accuracy = (1-(y_test_ejscreen - test_pred_ejscreen) ** 2).sum()/len(y_test_ejscreen)

    test_noejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] <= 0)
    y_test_noejscreen = y_test.iloc[test_noejscreen_indices]
    test_pred_noejscreen = test_pred[test_noejscreen_indices]
    noejscreen_accuracy = (1-(y_test_noejscreen - test_pred_noejscreen) ** 2).sum()/len(y_test_noejscreen)
    
    # calculate accuracy on positive areas (TP/(FN+TP))
    #overall_accuracy_pos = sum(np.logical_and(y_test == 1, test_pred == 1))/sum(y_test)
    ej_accuracy_pos = sum(np.logical_and(y_test == 1, \
                                         np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] > 0)))\
                                            /sum(np.logical_and(y_test == 1, X_test["EJSCREEN_FLAG_US_Y"] > 0))
    noej_accuracy_pos = sum(np.logical_and(y_test == 1, \
                                           np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0)))\
                                            /sum(np.logical_and(y_test == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0))
    ej_accuracy_neg = sum(np.logical_and(y_test == 0, \
                                         np.logical_and(test_pred == 0, X_test["EJSCREEN_FLAG_US_Y"] > 0)))\
                                            /sum(np.logical_and(y_test == 0, X_test["EJSCREEN_FLAG_US_Y"] > 0))
    noej_accuracy_neg = sum(np.logical_and(y_test == 0, \
                                           np.logical_and(test_pred == 0, X_test["EJSCREEN_FLAG_US_Y"] <= 0)))\
                                            /sum(np.logical_and(y_test == 0, X_test["EJSCREEN_FLAG_US_Y"] <= 0))
    
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
    pred_pos_ej_n = np.array(["n positive, ej", sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] > 0))])
    pred_pos_noej_n = np.array(["n positive, no ej", sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0))])
    #tosave = np.array([acc,ej_acc,noej_acc,acc_pos,\
    #                   ej_acc_pos,noej_acc_pos,pred_pos,\
    #                   pred_pos_ej,pred_pos_noej,ratio_ej_noej,true_ratio_ej_noej])
    tosave = np.array([balanced_acc_ej, balanced_acc_noej, pred_pos_ej_n, pred_pos_noej_n])
    if save:
        np.savetxt("outputs/" + name + ".txt", tosave, fmt='%s')

    #return np.mean(test_pred_ejscreen)/np.mean(test_pred_noejscreen),\
        #np.mean(y_test_ejscreen)/np.mean(y_test_noejscreen)
    return sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] >0))/sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0))

# for an already fit model
def summarize_model(X_train, y_train, X_test, y_test, model, name, save = True):

    predicted_probs = model.predict_proba(X_train)

    # get optimal threshold
    preds = predicted_probs[:,1]
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_train, preds)
    #roc_auc = sklearn.metrics.auc(fpr, tpr)
    threshold_opt = threshold[np.argmax(tpr - fpr)] 

    # save ROC curve with optimal threshold point
    if save:
        plt.clf()
        plt.plot(fpr, tpr)
        plt.plot([fpr[np.argmax(tpr - fpr)]],[tpr[np.argmax(tpr - fpr)]],'o')
        plt.savefig("outputs/" + name + '.png')

    # run on test data
    test_pred = (model.predict_proba(X_test)[:,1] >= threshold_opt).astype(bool)
    sklearn.metrics.confusion_matrix(y_test, test_pred)
    #print(sklearn.metrics.classification_report(y_test, test_pred))

    # accuracy overall
    overall_accuracy = (1-(y_test - test_pred) ** 2).sum()/len(y_test)

    # check proportion of test facilities it gets right in and outside of EJSCREEN areas
    test_ejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] >0)
    y_test_ejscreen = y_test.iloc[test_ejscreen_indices]
    test_pred_ejscreen = test_pred[test_ejscreen_indices]
    ejscreen_accuracy = (1-(y_test_ejscreen - test_pred_ejscreen) ** 2).sum()/len(y_test_ejscreen)

    test_noejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] <= 0)
    y_test_noejscreen = y_test.iloc[test_noejscreen_indices]
    test_pred_noejscreen = test_pred[test_noejscreen_indices]
    noejscreen_accuracy = (1-(y_test_noejscreen - test_pred_noejscreen) ** 2).sum()/len(y_test_noejscreen)
    
    # calculate accuracy on positive areas (TP/(FN+TP))
    #overall_accuracy_pos = sum(np.logical_and(y_test == 1, test_pred == 1))/sum(y_test)
    ej_accuracy_pos = sum(np.logical_and(y_test == 1, \
                                         np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] >0)))\
                                            /sum(np.logical_and(y_test == 1, X_test["EJSCREEN_FLAG_US_Y"] >0))
    noej_accuracy_pos = sum(np.logical_and(y_test == 1, \
                                           np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0)))\
                                            /sum(np.logical_and(y_test == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0))
    ej_accuracy_neg = sum(np.logical_and(y_test == 0, \
                                         np.logical_and(test_pred == 0, X_test["EJSCREEN_FLAG_US_Y"] >0)))\
                                            /sum(np.logical_and(y_test == 0, X_test["EJSCREEN_FLAG_US_Y"] >0))
    noej_accuracy_neg = sum(np.logical_and(y_test == 0, \
                                           np.logical_and(test_pred == 0, X_test["EJSCREEN_FLAG_US_Y"] <= 0)))\
                                            /sum(np.logical_and(y_test == 0, X_test["EJSCREEN_FLAG_US_Y"] <= 0))
    
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
    pred_pos_ej_n = np.array(["n positive, ej", sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] >0))])
    pred_pos_noej_n = np.array(["n positive, no ej", sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0))])
    #tosave = np.array([acc,ej_acc,noej_acc,acc_pos,\
    #                   ej_acc_pos,noej_acc_pos,pred_pos,\
    #                   pred_pos_ej,pred_pos_noej,ratio_ej_noej,true_ratio_ej_noej])
    tosave = np.array([balanced_acc_ej, balanced_acc_noej, pred_pos_ej_n, pred_pos_noej_n])
    if save:
        np.savetxt("outputs/" + name + ".txt", tosave, fmt='%s')

    #return np.mean(test_pred_ejscreen)/np.mean(test_pred_noejscreen),\
        #np.mean(y_test_ejscreen)/np.mean(y_test_noejscreen)
    return sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] >0))/sum(np.logical_and(test_pred == 1, X_test["EJSCREEN_FLAG_US_Y"] <= 0))


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
    return y_train_perm

