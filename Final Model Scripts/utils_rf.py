import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, chi2, f_regression, mutual_info_regression, f_classif, SelectKBest
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, log_loss
from sklearn.metrics import roc_curve, roc_auc_score, auc
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

np.random.seed(1000)
#
# Input: the noc_nswers data file
# output:
#     1. non aggrageted table with skill vectors and workshop numbers
#     2. table above with only one row for each noc-workshop
#     3. indiviual expert answers
#     4. aggrageted expert answers (portion of experts who said an answer for a noc-workshop)
#     5. x with with one row for each noc - workshop number removed
#     6. portion of expert answers for a noc

#OK we are making sure sure everything is properly adjusted to being on noc level

def data_proccess(file,discrete):
    data = pd.read_csv(file,index_col=['noc','noc_code','workshop.number'])
    data.sort_index(inplace=True)
    data.loc[data.share == 'remain constant','share'] = 'constant'

    x = data.drop(['absolute','share','Unnamed: 0'],axis=1) #making x data frame
    x['work_num'] = x.index.get_level_values(1) #making workshop number a variable as well as an index
    if discrete:
        x = np.round(x).astype(int)#round x to make discrete

    #one hot bode workshop number
    enc = LabelBinarizer()
    enc.fit([1,2,3,4,5,6])
    cat_work_num = enc.transform(x['work_num'])

    x.drop(['work_num'],axis=1,inplace=True)
    work_nums = pd.DataFrame(cat_work_num,
                         columns=['work_num_1','work_num_2','work_num_3','work_num_4','work_num_5','work_num_6'],
                        index=x.index)
    x = pd.concat([x,work_nums],axis=1)# we drop this set of vars when we don't want to include workshop number

    x_agg = x.drop_duplicates()#drop to the noc-workshop lvl (every skill vector and workshop is a unique row)

    x_noclvl = x_agg.drop(['work_num_1','work_num_2','work_num_3','work_num_4','work_num_5','work_num_6'],axis=1).droplevel(2).drop_duplicates()#dropping to the noc lvl

    y = pd.DataFrame({'non_binned': data['share'],#making not increase bin and not decrease bin
              'increase': data['share'].str.replace('constant','not_increase').str.replace('decrease','not_increase'),
              'decrease': data['share'].str.replace('constant','not_decrease').str.replace('increase','not_decrease')})


    y_agg = pd.DataFrame(data['share']).pivot_table(index = ['noc','noc_code','workshop.number'], columns = 'share', aggfunc = len).fillna(0)
    y_agg['sum'] = y_agg.sum(axis = 1)
    y_noclvl = y_agg.groupby(level=[0,1]).sum()
    y_agg.loc[:,y_agg.columns!='sum'] = y_agg.loc[:,y_agg.columns!='sum'].divide(y_agg['sum'],axis=0)
    y_noclvl.loc[:,y_noclvl.columns!='sum'] = y_noclvl.loc[:,y_noclvl.columns!='sum'].divide(y_noclvl['sum'],axis=0)



    return x, x_agg, y, y_agg, x_noclvl, y_noclvl

# returns the hyper-paramaters for either the classification or regression model

def init_params(model_type):
    if model_type == 'cat':
        params = {
         'criterion': 'gini',
         'max_features': 'auto',
         'min_samples_leaf': 8,
         'min_samples_split': 10,
         'n_estimators': 250,
         'n_jobs':-1
        }
    if model_type == 'reg':
        params = {
         'criterion': 'mse',
         'max_features': None,
         'min_samples_leaf': 1,
         'min_samples_split': 15,
         'n_estimators': 250,
         'n_jobs':-1
        }
    return params

def run_k_fold (x,y,params,index,binned,model_type):

    x = pd.DataFrame(x)

    rf = RandomForestClassifier(**params)
    n_trees = params['n_estimators']

    if model_type == 'reg':
        rf = RandomForestRegressor(**params)

    if binned:
        pred = np.zeros(x.shape[0])
    else:
        pred = np.zeros((x.shape[0],3))

    gkf = GroupKFold(5)
    nocs = index.get_level_values(0)
    for train_index, test_index in gkf.split(x,y,nocs):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf.fit(x_train,y_train)

        if model_type == 'reg':
            pred[test_index] = rf.predict(x_test)
        if model_type == 'pred_probs':
            if binned:
                pred[test_index] = rf.predict_proba(x_test)[:,0]
            else:
                pred[test_index] = rf.predict_proba(x_test)
        if model_type == 'tree_port':
            tree_pred = np.zeros((n_trees,len(test_index)))
            for tree in range(n_trees):
                tree_pred[tree] = rf.estimators_[tree].predict(x_test)
            pred[test_index] = tree_pred.mean(axis=0)

    pred = pd.DataFrame(pred,index=index).groupby(level=0).mean()#this is grouping by only nocs. Switch back if you want workshops

    return pred

# def k_fold_feature_importance(x,y,model_type):
#     rf = RandomForestClassifier(**init_params(model_type))
#     kf = KFold(n_splits=10,shuffle=False)
#
#     if model_type == 'reg':
#         rf = RandomForestRegressor(**init_params(model_type))
#         kf = KFold(n_splits=5,shuffle=True)
#
#     feature_imp = np.zeros((x.shape[1],5))
#     i=0
#
#     for train_index, test_index in kf.split(x):
#         x_train, x_test = x.iloc[train_index], x.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#         rf.fit(x_train,y_train)
#
#         feature_imp[:,i] = rf.feature_importances_
#
#     return feature_imp.mean(axis=1)

def param_search(x,y,model_type):

    param_grid= {'n_estimators':[100,150,250,275,300,600,1000],#number of trees
             'min_samples_leaf': [1,2,4,8],#minimum number of data points can be used to make a leaf at the end of a tree
             'min_samples_split': [5,10,15]#min number of data points to split a branch
             }

    gkf=GroupKFold(n_splits=5)

    if model_type == 'reg':
        rf = RandomForestRegressor(**init_params(model_type))
        search = GridSearchCV(rf,param_grid,scoring='neg_mean_squared_error',cv=gkf,n_jobs=-1,iid=False)

    if model_type == 'cat':#switched scorer to custom MAE though I'm not entirelly sure it'll work in this context. It should though
        rf = RandomForestClassifier(**init_params(model_type))
        param_grid['criterion'] = ['gini','entropy']
        search = GridSearchCV(rf,param_grid,scoring='roc_auc',cv=gkf,n_jobs=-1,iid=False)

    search.fit(x,y,x.index.get_level_values(0))

    return search.best_params_, search.cv_results_


def basic_feature_selection(x,y,model_type,k):
    if model_type == 'class':
        return SelectKBest(mutual_info_classif,k).fit_transform(x,y)

    if model_type == 'reg':
        return SelectKBest(mutual_info_regression,k).fit_transform(x,y)


#Feature scores for a number of differnt measures
def different_feature_rankings(x, x_agg, y, y_agg):
    mi_class, chi2_class, f_class = basic_feature_selection(x,y['increase'],'class')
    mi_reg, f_reg = basic_feature_selection(x_agg,y_agg['increase'],'reg')

    feature_scores = pd.DataFrame({'mi_class': mi_class.scores_,
                 'chi2': chi2_class.scores_,
                 'f_class': f_class.scores_,
                 'mi_reg': mi_reg.scores_,
                 'f_reg': f_reg.scores_},index=x.columns)

    feature_scores.sort_values('mi_reg',ascending = False)

def scores_by_k(x, x_agg, y, y_agg):
    reg_scores = np.zeros((120,120))
    class_scores = np.zeros((120,120))

    for k in range(1,121):
        reg_scores[k-1] = run_k_fold(basic_feature_selection(x_agg,y_agg['increase'],'reg',k),
                            y_agg['increase'],
                            init_params('reg'),
                            x_agg.index,
                            True,'reg').iloc[:,0].values
        class_scores[k-1] = run_k_fold(basic_feature_selection(x,y['increase'],'class',k),
                            y['increase'],
                            init_params('cat'),
                            x.index,
                            True,'pred_probs').iloc[:,0].values

    r_scores = abs(pd.DataFrame(reg_scores,index=range(1,121),columns=x_agg.index).T.subtract(y_agg['increase'],axis=0)).mean(axis=0)
    c_scores = abs(pd.DataFrame(class_scores,index=range(1,121),columns=x_agg.index).T.subtract(y_agg['increase'],axis=0)).mean()

    return r_scores, c_scores

def run_models(x, x_agg, y, y_agg,binned,increase,k_reg,k_class):#this all needs to be reqorked

    if binned:
        if increase:
            #set number of features to run with
            # x_agg_cut = basic_feature_selection(x_agg,y_agg['increase'],'reg',k_reg)
            # x_cut = basic_feature_selection(x,y['increase'],'class',k_class)

            #run regression in a k-fold framework and place results into dataframes
            pred = pd.concat([
                run_k_fold(x_agg_cut,y_agg['increase'],init_params('reg'),x_agg.index,binned,'reg'),
                run_k_fold(x_cut,y['increase'],init_params('cat'),x.index,binned,'pred_probs'),
                run_k_fold(x_cut,y['increase'],init_params('cat'),x.index,binned,'tree_port')
            ],axis=1)

        else:
            # x_agg_cut = basic_feature_selection(x_agg,y_agg['decrease'],'reg',k_reg)
            # x_cut = basic_feature_selection(x,y['decrease'],'class',k_class)

            #run regression in a k-fold framework and place results into dataframes
            pred = pd.concat([
                run_k_fold(x_agg_cut,y_agg['decrease'],init_params('reg'),x_agg.index,binned,'reg'),
                run_k_fold(x_cut,y['decrease'],init_params('cat'),x.index,binned,'pred_probs'),
                run_k_fold(x_cut,y['decrease'],init_params('cat'),x.index,binned,'tree_port')
            ],axis=1)

    else:
        y_agg = y_agg[['constant','decrease','increase']]
        y = y['non_binned']
        pred = pd.concat([
            run_k_fold(x_agg,y_agg,init_params('reg'),x_agg.index,binned,'reg'),
            run_k_fold(x,y,init_params('cat'),x.index,binned,'pred_probs')#,
            #run_k_fold(x,y,init_params('cat'),x.index,binned,'tree_port')
        ],axis=1)


    pred.set_index(x_agg.index.get_level_values(0).drop_duplicates(),inplace = True)
    if binned:
        pred.columns = ['regression','pred_prob','tree_portions']
    else:
        pred.columns = ['regression_con','regression_dec','regression_inc',
                                 'prob_con','prob_dec','prob_inc']

    return pred

def confusion_matrix(pred,truth,type):
    if type==1:
        matrix = pd.DataFrame(
         [[sum(np.logical_and(truth>=0.5,pred>=0.5)),
           sum(np.logical_and(truth>=0.5,pred<0.5))],
          [sum(np.logical_and(truth<0.5,pred>=0.5)),
           sum(np.logical_and(truth<0.5,pred<0.5))]]
        ,columns=['pred_increase','pred_decrease'],index=['true_increase','true_decrease'])
    if type==2:
        pred.columns = ['constant','decrease','increase']
        matrix = pd.DataFrame(
        [[sum(np.logical_and(truth.idxmax(axis=1)=='increase',pred.idxmax(axis=1)=='increase')),
          sum(np.logical_and(truth.idxmax(axis=1)=='increase',pred.idxmax(axis=1)=='constant')),
          sum(np.logical_and(truth.idxmax(axis=1)=='increase',pred.idxmax(axis=1)=='decrease'))],
         [sum(np.logical_and(truth.idxmax(axis=1)=='constant',pred.idxmax(axis=1)=='increase')),
          sum(np.logical_and(truth.idxmax(axis=1)=='constant',pred.idxmax(axis=1)=='constant')),
          sum(np.logical_and(truth.idxmax(axis=1)=='constant',pred.idxmax(axis=1)=='decrease'))],
         [sum(np.logical_and(truth.idxmax(axis=1)=='decrease',pred.idxmax(axis=1)=='increase')),
          sum(np.logical_and(truth.idxmax(axis=1)=='decrease',pred.idxmax(axis=1)=='constant')),
          sum(np.logical_and(truth.idxmax(axis=1)=='decrease',pred.idxmax(axis=1)=='decrease'))]],
        columns=['pred_increase','pred_constant','pred_decrease'],index=['true_increase','true_constant','true_decrease'])
    if type==3:
        matrix = pd.DataFrame(
         [[sum(np.logical_and(truth>=0.7,pred>=0.7)),
           sum(np.logical_and(truth>=0.7,np.logical_and(pred>=0.5,pred<0.7))),
           sum(np.logical_and(truth>=0.7,np.logical_and(pred>=0.3,pred<0.5))),
           sum(np.logical_and(truth>=0.7,pred<0.3))],
          [sum(np.logical_and(np.logical_and(truth>=0.5,truth<0.7),pred>=0.7)),
           sum(np.logical_and(np.logical_and(truth>=0.5,truth<0.7),np.logical_and(pred>=0.5,pred<0.7))),
           sum(np.logical_and(np.logical_and(truth>=0.5,truth<0.7),np.logical_and(pred>=0.3,pred<0.5))),
           sum(np.logical_and(np.logical_and(truth>=0.5,truth<0.7),pred<0.3))],
          [sum(np.logical_and(np.logical_and(truth>=0.3,truth<0.5),pred>=0.7)),
           sum(np.logical_and(np.logical_and(truth>=0.3,truth<0.5),np.logical_and(pred>=0.5,pred<0.7))),
           sum(np.logical_and(np.logical_and(truth>=0.3,truth<0.5),np.logical_and(pred>=0.3,pred<0.5))),
           sum(np.logical_and(np.logical_and(truth>=0.3,truth<0.5),pred<0.3))],
          [sum(np.logical_and(truth<0.3,pred>=0.7)),
           sum(np.logical_and(truth<0.3,np.logical_and(pred>=0.5,pred<0.7))),
           sum(np.logical_and(truth<0.3,np.logical_and(pred>=0.3,pred<0.5))),
           sum(np.logical_and(truth<0.3,pred<0.3))]]
        ,columns=['pred>=0.7','0.5<=pred<0.7','0.3<=pred<0.5','pred<0.3'],index=['truth>=0.7','0.5<=truth<0.7','0.3<=truth<0.5','truth<0.3'])
    return matrix

def run_sfs(x,y,model_type,custom_score,increase_model):
    nocs = x.index.get_level_values(0)
    cv_gen = GroupKFold(5).split(x, y, nocs)
    cv = list(cv_gen)

    if model_type == 'reg':
        rf = RandomForestRegressor(**init_params(model_type))
        sfs = SFS(rf,
           k_features=(1,20),
           forward=True,
           floating=True,
           verbose=2,
           scoring='neg_mean_squared_error',
           cv=cv,
           n_jobs=-1)
    else:
        if custom_score:
            if increase_model:
                scorer=make_scorer(custom_MAE_increase,greater_is_better=False,needs_proba=True)
            else:
                scorer=make_scorer(custom_MAE_decrease,greater_is_better=False,needs_proba=True)
        else:
            scorer='roc_auc'

        rf = RandomForestClassifier(**init_params(model_type))
        sfs = SFS(rf,
           k_features=(1,20),
           forward=True,
           floating=True,
           verbose=2,
           scoring=scorer,
           cv=cv,
           n_jobs=-1)

    sfs.fit(x,y)

    return sfs

def custom_MAE_increase(y_true, y_pred):
    totals = y_true.groupby(level=0).count()#aggregates on the noc level, can switch back to noc-workshop if needed
    increase_count = y_true[y_true=='increase'].groupby(level=0).count()
    y_true_agg = increase_count.divide(totals).fillna(0)

    y_true_agg_rep = np.repeat(y_true_agg[0],totals[0])
    for i in range(1,len(y_true_agg)):#this is kind of a dumb way of doing this. might fix later
        y_true_agg_rep = np.concatenate((y_true_agg_rep,np.repeat(y_true_agg[i],totals[i])),axis=None)

    return mean_absolute_error(y_true_agg_rep,y_pred)

def custom_MAE_decrease(y_true, y_pred):
    totals = y_true.groupby(level=0).count()#aggregates on the noc level, can switch back to noc-workshop if needed
    increase_count = y_true[y_true=='decrease'].groupby(level=0).count()
    y_true_agg = increase_count.divide(totals).fillna(0)

    y_true_agg_rep = np.repeat(y_true_agg[0],totals[0])
    for i in range(1,len(y_true_agg)):#this is kind of a dumb way of doing this. might fix later
        y_true_agg_rep = np.concatenate((y_true_agg_rep,np.repeat(y_true_agg[i],totals[i])),axis=None)

    return mean_absolute_error(y_true_agg_rep,y_pred)
