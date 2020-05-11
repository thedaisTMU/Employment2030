import rf_scripts.utils_rf as urf
import pickle

file = "../tables/noc_answers.csv"
x, x_agg, y, y_agg, x_noclvl, y_noclvl = urf.data_proccess(file,discrete=True)

mae_results = []

for i in range(20):
    sfs_mae = urf.run_sfs(x,y['increase'],'cat',True)

    mae_results.append(sfs_mae.get_metric_dict())

with open('mae_results_ws_cat2.pkl','wb') as f:
    pickle.dump(mae_results,f)
