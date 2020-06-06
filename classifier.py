"""
Classify users and trolls based on their estimated rewards.

It takes as input two csv files that contains the rewards estimated with main_IRL.py:
- df_results_users_IRL.csv
- df_results_trolls_IRL.csv

It returns:
- classification performance
- feature importance
- estimated rewards and features weights

Luca Luceri, 2020
luca.luceri@supsi.ch
"""



import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
# Set random seed
np.random.seed(0)
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import sklearn.metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


################################## functions ##################################

# dataframe of estimated rewards
def estimated_rewards(df):
    df_f = pd.DataFrame(columns=['NT+nt','NT+rp','NT+rt','NT+tw','RP+nt','RP+rp','RP+rt','RP+tw','RT+nt','RT+rp','RT+rt','RT+tw'])
    for i in np.arange(len(df)):
        f_i = df.iloc[i]["r"]
        f_i=f_i[1:]
        f_i = f_i[:-1]
        f_u = np.fromstring(f_i, dtype=float, sep=' ')
        row = [f_u[0],f_u[1],f_u[2],f_u[3],f_u[4],f_u[5],f_u[6],f_u[7],f_u[8],f_u[9],f_u[10],f_u[11]]
        df_f.loc[i]=row
    return df_f

# dataframe of estimated weights
def estimated_weights(df):
    df_w = pd.DataFrame(columns=["RT","RP","tw","rt","rp"])
    for i in np.arange(len(df)):
        w_i = df.iloc[i]["w"]
        w_i=w_i[1:]
        w_i = w_i[:-1]
        w_u = np.fromstring(w_i, dtype=float, sep=' ')
        row = [w_u[0],w_u[1],w_u[2],w_u[3],w_u[4]]
        df_w.loc[i]=row
    return df_w


# features based on estimated rewards
def build_rewards_only_df(df_users,df_trolls):
    df_f_u = estimated_rewards(df_users)
    df_f_u['label']=0 # label 0 for users / label 1 for trolls
    df_f_t = estimated_rewards(df_trolls)
    df_f_t['label']=1 # label 0 for users / label 1 for trolls
    df_f = df_f_u.append(df_f_t,ignore_index=True)
    seed = 7
    np.random.seed(seed)
    df_f=df_f.sample(frac=1).reset_index(drop=True) 
    X_df = df_f.copy()
    del X_df['label']
    X= X_df.values
    Y = df_f['label'].values
    return df_f_u, df_f_t,X,Y

# cross-validation
def machine_learning_prediction_CM(X,Y,folds,classifier):
    A=[]
    P=[]
    R=[]
    F=[]
    fi=[]
    AUC = []
    TPR = []
    TNR = []
    seed = 7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in kfold.split(X, Y):
        clf = classifier 
        clf.fit(X[train], Y[train])
        check_predictions = clf.predict(X[test])
        check_lab = Y[test]
        prob_predictions = clf.predict_proba(X[test])
        fpr, tpr, thresholds = metrics.roc_curve(check_lab, prob_predictions[:,1])
        auc=metrics.auc(fpr, tpr)
        acc=sklearn.metrics.accuracy_score(check_lab, check_predictions)
        prec = sklearn.metrics.precision_score(check_lab, check_predictions)
        rec = sklearn.metrics.recall_score(check_lab, check_predictions)
        f1 = sklearn.metrics.f1_score(check_lab, check_predictions)
        tn, fp, fn, tp = confusion_matrix(check_lab,check_predictions).ravel()
        tpr = tp/float(tp+fn)
        tnr = tn/float(tn+fp)
        TPR.append(tpr)
        TNR.append(tnr)
        A.append(acc)
        P.append(prec)
        R.append(rec)
        F.append(f1)
        fi.append(clf.feature_importances_)
        AUC.append(auc)

    return (A,P,R,F,fi,AUC,TPR,TNR)



########################################################################################################################################

# load rewards for trolls and users
df_users_=pd.read_csv("df_results_users_IRL.csv")
df_trolls_=pd.read_csv("df_results_trolls_IRL.csv")

# train and test model
folds = 10 # number of folds for cross-validation
split=5 # undersampling techniques (users set is 5 times larger than trolls set)

# classifier instance
classifier = AdaBoostClassifier(n_estimators=500,learning_rate = 0.05) 

#metrics
Acc =[]
Pre =[]
Rec = []
Auc = []
F1 =[]
FI =[]
TPR = []
TNR = []

for i in np.arange(split):
	# undersampling
    start=int(i*np.ceil(float(len(df_users_))/split))
    end =int((i+1)*np.ceil(float(len(df_users_))/split))
    df_bal=df_users_[start:end]
    df_f_u, df_f_t,X,Y=build_rewards_only_df(df_bal,df_trolls_) # capture rewards estimated via IRL
    A,P,R,F,fi,AUC,tpr,tnr = machine_learning_prediction_CM(X,Y,folds,classifier) # cross-validation
    Acc.append(A)
    Pre.append(P)
    Rec.append(R)
    F1.append(F)
    Auc.append(AUC)
    FI.append(fi)
    TPR.append(tpr)
    TNR.append(tnr)
    

print("IRL-based classification")
print("accuracy",np.mean(Acc),np.std(Acc))
print("precision",np.mean(Pre),np.std(Pre))
print("recall",np.mean(Rec),np.std(Rec))
print("f1",np.mean(F1),np.std(F1))
print("AUC",np.mean(Auc),np.std(Auc))
print("TPR",np.mean(TPR),np.std(TPR))
print("TNR",np.mean(TNR),np.std(TNR))


#### feature importance
fi_split_mean = []
fi_split_std = []
for i in np.arange(len(FI)):
    fi_split = FI[i]
    fi_split_mean.append(np.mean(fi_split,axis=0))
    fi_split_std.append(np.std(fi_split,axis=0)) 

features = df_f_u.columns
importances = np.mean(fi_split_mean,axis=0)
std =np.std(fi_split_mean,axis=0)
indices = np.argsort(importances)
plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r',xerr=std[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


##### estimated rewards for users and trolls
df_f_u = estimated_rewards(df_users_)
df_f_t = estimated_rewards(df_trolls_)

# estimated feature weights for users and trolls
df_w_u = estimated_weights(df_users_)
df_w_t = estimated_weights(df_trolls_)

