# running logistic regression baseline
# conda install --channel=conda-forge scikit-learn
# https://realpython.com/logistic-regression-python/#multi-variate-logistic-regression

# libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas
import random
import sys
sys.path.append('code')
from baseline_analysis_functions import run_model, make_permuted_training

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

##############################
##### NO PERMUTE DATA ##############################
##############################
#run_model(X_train, y_train, X_test, y_test, "perm_0")

##############################
##### PERMUTE DATA ##############################
##############################

x_vals = np.linspace(start=0, stop=.75, num=20)
models = [LogisticRegression(solver='liblinear', random_state=229, max_iter=1000)]
          #LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, random_state=229, max_iter=1000),\
          #LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)]
model_labs = ["Logistic Regression"]
ratios = np.zeros((len(x_vals), len(models)))

for i in np.arange(len(x_vals)):
    for j in np.arange(len(models)):
        ratio = run_model(X_train, make_permuted_training(X_train, y_train, x_vals[i]), X_test, y_test, \
                                models[j], "perm_"+str(x_vals[i]), save = False)
        ratios[i,j]=ratio

test_noejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] == 0)
y_test_noejscreen = sum(y_test.iloc[test_noejscreen_indices])
test_ejscreen_indices = np.where(X_test["EJSCREEN_FLAG_US_Y"] == 1)
y_test_ejscreen = sum(y_test.iloc[test_ejscreen_indices])

plt.clf()
for j in np.arange(len(models)):
    plt.plot(x_vals, ratios[:,j], label = model_labs[j])
    plt.axhline(y_test_ejscreen/y_test_noejscreen, linestyle='--', color= 'orange')
plt.xlabel("Proportion of low-quality inspections in EJ areas")
plt.ylabel("Number of predicted violations in EJ areas/non-EJ areas")
plt.legend(loc='best')
plt.savefig("outputs/ratios.png")

model = LogisticRegression(solver='liblinear', random_state=229, max_iter=1000)
run_model(X_train, make_permuted_training(X_train, y_train, 0), X_test, y_test, \
              model, "perm_0", save = True)
model = LogisticRegression(solver='liblinear', random_state=229, max_iter=1000)
run_model(X_train, make_permuted_training(X_train, y_train, .1), X_test, y_test, \
              model, "perm_.1", save = True)
model = LogisticRegression(solver='liblinear', random_state=229, max_iter=1000)
run_model(X_train, make_permuted_training(X_train, y_train, .3), X_test, y_test, \
              model, "perm_.3", save = True)
    