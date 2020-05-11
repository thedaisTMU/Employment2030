import rf_scripts.utils_rf as urf
import pickle

file = "../tables/noc_answers.csv"
x, x_agg, y, y_agg, x_noclvl, y_noclvl = urf.data_proccess(file,discrete=True)

x.drop(['work_num_1','work_num_2','work_num_3','work_num_4','work_num_5','work_num_6'],axis=1,inplace=True)

mae_results = []

for i in range(20):
    sfs_mae = urf.run_sfs(x,y['decrease'],'cat',custom_score=True,increase_model=False)
    mae_results.append(sfs_mae.get_metric_dict())

with open('mae_results_dec2.pkl','wb') as f:
    pickle.dump(mae_results,f)
