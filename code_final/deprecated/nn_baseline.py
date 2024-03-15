from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas
import random
import sys
sys.path.append('code')
from baseline_analysis_functions import run_model, make_permuted_training, summarize_model
from sklearn.preprocessing import StandardScaler  
from sklearn.compose import ColumnTransformer

random.seed(229)

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
random_full_x = pandas.get_dummies(random_full[x_names])
random_full_y = random_full["DV"]

# make train/test split
X_train, X_test, y_train, y_test = train_test_split(random_full_x,random_full_y,\
                                                    test_size=0.20,random_state=229)

#### SCALING OPTION 1 ######################################
# create columns
colnames = X_train.columns.values
colnames = np.delete(colnames, np.where(colnames == 'EJSCREEN_FLAG_US_Y'))
colnames = np.append(colnames, 'EJSCREEN_FLAG_US_Y')

# scale data
scaler = StandardScaler() 
mycols = X_train.columns.values
mycols = np.delete(mycols, np.where(mycols=='EJSCREEN_FLAG_US_Y'))
ct = ColumnTransformer([
        ('somename', StandardScaler(), mycols)
    ], remainder='passthrough')

X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)
X_train = pandas.DataFrame(X_train, columns = colnames)
X_test = pandas.DataFrame(X_test, columns = colnames)
X_train.index = y_train.index
X_test.index = y_test.index

#### SCALING OPTION 2 ######################################
colnames = X_train.columns.values
# fit scaler on training data
scaler = StandardScaler() 
scaler.fit(X_train)  
X_train[colnames] = scaler.transform(X_train[colnames])  
# apply same transformation to test data
X_test[colnames] = scaler.transform(X_test[colnames]) 
X_train = pandas.DataFrame(X_train, columns = colnames)
X_test = pandas.DataFrame(X_test, columns = colnames)
X_train.index = y_train.index
X_test.index = y_test.index

#https://stackoverflow.com/questions/70007916/how-to-scale-all-columns-except-certain-ones-in-pandas-dataframe 
#cols = X_train.columns[X_train.columns != 'EJSCREEN_FLAG_US_Y']
##X_train[cols] = scaler.fit_transform(X_train[cols])

# fit scaler on training data
#scaler.fit(X_train)  
#X_train[cols] = scaler.transform(X_train[cols])  
# apply same transformation to test data
#X_test[cols] = scaler.transform(X_test[cols])  

# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# start with 2 layer, width 128
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(32, 32), activation='relu',
#                     random_state=229)
#clf.fit(X_train, y_train)


x_vals = np.linspace(start=0, stop=.75, num=3)
models = [MLPClassifier(solver='adam', 
                     hidden_layer_sizes=(128, 128), activation='relu',
                     random_state=229, max_iter = 1000)]
model_labs = ["2-layer, 128 node MLP (ReLu)"]
ratios = np.zeros((len(x_vals), len(models)))

for i in np.arange(len(x_vals)):
    for j in np.arange(len(models)):
        print(x_vals[i])
        ratio = run_model(X_train, make_permuted_training(X_train, y_train, x_vals[i]), X_test, y_test, \
                                models[j], "perm_"+str(x_vals[i]), save = False)
        ratios[i,j]=ratio

test_noejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] <= 0)
y_test_noejscreen = sum(y_test.iloc[test_noejscreen_indices])
test_ejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] >0)
y_test_ejscreen = sum(y_test.iloc[test_ejscreen_indices])
X_train[y_train == 1]
plt.clf()
for j in np.arange(len(models)):
    plt.plot(x_vals, ratios[:,j], label = model_labs[j])
    plt.axhline(y_test_ejscreen/y_test_noejscreen, linestyle='--', color= 'orange')
plt.xlabel("Proportion of low-quality inspections in EJ areas")
plt.ylabel("Number of predicted violations in EJ areas/non-EJ areas")
plt.legend(loc='best')
plt.savefig("outputs/nn_2layer_128.png")

################### EARLY STOPPING ####################
x_vals = np.linspace(start=0, stop=.75, num=3)
models = [MLPClassifier(solver='adam', 
                     hidden_layer_sizes=(128, 128), activation='relu',
                     random_state=229, max_iter = 1000,
                     early_stopping = True)]
model_labs = ["2-layer, 128 node MLP (ReLu)"]
ratios = np.zeros((len(x_vals), len(models)))

for i in np.arange(len(x_vals)):
    for j in np.arange(len(models)):
        print(x_vals[i])
        ratio = run_model(X_train, make_permuted_training(X_train, y_train, x_vals[i]), X_test, y_test, \
                                models[j], "perm_"+str(x_vals[i]), save = False)
        ratios[i,j]=ratio

test_noejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] <= 0)
y_test_noejscreen = sum(y_test.iloc[test_noejscreen_indices])
test_ejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] >0)
y_test_ejscreen = sum(y_test.iloc[test_ejscreen_indices])
X_train[y_train == 1]
plt.clf()
for j in np.arange(len(models)):
    plt.plot(x_vals, ratios[:,j], label = model_labs[j])
    plt.axhline(y_test_ejscreen/y_test_noejscreen, linestyle='--', color= 'orange')
plt.xlabel("Proportion of low-quality inspections in EJ areas")
plt.ylabel("Number of predicted violations in EJ areas/non-EJ areas")
plt.legend(loc='best')
plt.savefig("outputs/nn_2layer_128_earlystop.png")
