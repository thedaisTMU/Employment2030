mport numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
np.random.seed(1000)

#reading the data
file = "../tables/scores_answers.csv"

data = pd.read_csv(file,index_col=['noc','workshop.number'])
data.sort_index(inplace=True)
data.loc[data.share == 'remain constant','share'] = 'constant'
data.dropna(inplace=True)


#splitting up x and y
x = data.drop(['absolute','share','Unnamed: 0','noc_code'],axis=1) #making x data frame
x['work_num'] = x.index.get_level_values(1) #making workshop number a variable as well as an index
x.drop_duplicates(inplace=True)
x = np.round(x).astype(int)#round x to make discrete

#creating y variables
y_abs = pd.DataFrame(data['absolute']).pivot_table(index = ['noc','workshop.number'], columns = 'absolute', aggfunc = len).fillna(0)
y_abs['sum'] = y_abs.sum(axis = 1)
y_abs['not_increase'] = y_abs['fewer'] + y_abs['same']
y_abs.loc[:,y_abs.columns!='sum'] = y_abs.loc[:,y_abs.columns!='sum'].divide(y_abs['sum'],axis=0)
y_abs['y'] = y_abs[['fewer','more','same']].idxmax(axis=1)
y_abs['binned_y'] = y_abs[['more','not_increase']].idxmax(axis=1)

y_share = pd.DataFrame(data['share']).pivot_table(index = ['noc','workshop.number'], columns = 'share', aggfunc = len).fillna(0)
y_share['sum'] = y_share.sum(axis = 1)
y_share['not_increase'] = y_share['decrease'] + y_share['constant']
y_share.loc[:,y_share.columns!='sum'] = y_share.loc[:,y_share.columns!='sum'].divide(y_share['sum'],axis=0)
y_share['y'] = y_share[['constant','decrease','increase']].idxmax(axis=1)
y_share['binned_y'] = y_share[['increase','not_increase']].idxmax(axis=1)

y = [y_abs['y'],y_share['y'],
     y_abs['binned_y'],y_share['binned_y'],
     y_abs[['more','same']],y_share[['increase','constant']],
     y_abs['more'],y_share['increase']]

x_array = np.asarray(x)

rf =[ #share_bin,share_cont,abs_bin_cont,share_bin_cont
RandomForestClassifier(criterion = 'gini',class_weight='balanced',max_features='auto',min_samples_leaf=4,min_samples_split=10, n_estimators=2000,n_jobs=-1),
RandomForestRegressor(criterion='mse',max_features=None,min_samples_leaf=10, min_samples_split=15,n_estimators=2000,n_jobs=-1),
RandomForestRegressor(criterion='mse',max_features=None,min_samples_leaf=2, min_samples_split=5,n_estimators=2000,n_jobs=-1)
]

sfs_rf = [
SFS(estimator = rf[0],k_features=(30,90),forward=False,floating=True,scoring='accuracy',n_jobs=-1,verbose=1),
SFS(estimator = rf[2],k_features=(30,90),forward=False,floating=True,scoring=make_scorer(mean_squared_error),n_jobs=-1,verbose=1)
]

#running july 30th
sfs_rf[1].fit(x_array,y[7])

with open('SFS_share_binned.txt', 'w') as f:
    f.write(str(sfs_rf[1].k_score_))
    f.write(str(sfs_rf[1].k_feature_idx_))
