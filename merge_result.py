import os
from glob import glob
import pandas as pd

base_dir = '/mnt/home/linjie/projects/solubility/result/classification/CYP_hERG_clean_data'
dataset_names = os.listdir(base_dir)
dataset_names.remove('run_all.py')
model_names = ['AutoGluonModel', 'CMPNN', 'DMPNN', 'GraphSAGE', 'GraphNet', 'DGCNN', 'ECC', 'GAT']
for dataset_name in dataset_names:
    print(dataset_name)
    columns = pd.read_csv(os.path.join(base_dir, dataset_name, 'CMPNN', 'performance.csv')).columns
    df_all = pd.DataFrame(columns=list(columns))
    for model in model_names:
        if model in ['AutoGluonModel', 'CMPNN', 'DMPNN']:
            test_score_path = os.path.join(base_dir, dataset_name, model, 'performance.csv')
        else:
            test_score_path = os.path.join(base_dir, dataset_name, model, 'inner_5', 'performance.csv')
        test_score_df = pd.read_csv(test_score_path)
        df_new = test_score_df[columns]
        df_all = pd.concat([df_all, df_new], axis=0,ignore_index=True)
    df_all['dataset'] = dataset_name
    df_all = df_all.reindex(columns=['dataset']+list(columns))
    df_all.to_csv(os.path.join(base_dir, dataset_name, 'metric_'+dataset_name+'.csv'), index=False, float_format='%.3f')
    print(df_all)




base_dir = '/mnt/home/linjie/projects/solubility/result/classification/modelling_data_1'
dataset_names = os.listdir(base_dir)
dataset_names = list(set(dataset_names)-set(['run_all.py', 'F_data', 'MDCK_ER', 'PAMPA_permeability']))
model_names = ['AutoGluonModel', 'CMPNN', 'DMPNN', 'GraphSAGE', 'GraphNet', 'DGCNN', 'ECC', 'GAT']
for dataset_name in ['F_data']:
    print(dataset_name)
    columns = pd.read_csv(os.path.join(base_dir, dataset_name, 'CMPNN', 'performance.csv')).columns
    df_all = pd.DataFrame(columns=list(columns))
    for model in model_names:
        if model in ['AutoGluonModel', 'CMPNN', 'DMPNN']:
            test_score_path = os.path.join(base_dir, dataset_name, model, 'performance.csv')
        else:
            test_score_path = os.path.join(base_dir, dataset_name, model, 'inner_5', 'performance.csv')
        test_score_df = pd.read_csv(test_score_path)
        df_new = test_score_df[columns]
        df_all = pd.concat([df_all, df_new], axis=0,ignore_index=True)
    df_all['dataset'] = dataset_name
    df_all = df_all.reindex(columns=['dataset']+list(columns))
    df_all.to_csv(os.path.join(base_dir, dataset_name, 'metric_'+dataset_name+'.csv'), index=False, float_format='%.3f')
    print(df_all)



base_dir = '/mnt/home/linjie/projects/solubility/result/regression/Gostar_LogD_Solubility'
dataset_names = os.listdir(base_dir)
dataset_names = list(set(dataset_names)-set(['run_all.py']))
model_names = ['AutoGluonModel', 'CMPNN', 'DMPNN', 'GraphSAGE', 'GraphNet', 'DGCNN', 'ECC', 'GAT']
for dataset_name in dataset_names:
    print(dataset_name)
    columns = pd.read_csv(os.path.join(base_dir, dataset_name, 'CMPNN', 'performance.csv')).columns
    df_all = pd.DataFrame(columns=list(columns))
    for model in model_names:
        if model in ['AutoGluonModel', 'CMPNN', 'DMPNN']:
            test_score_path = os.path.join(base_dir, dataset_name, model, 'performance.csv')
        else:
            test_score_path = os.path.join(base_dir, dataset_name, model, 'inner_5', 'performance.csv')
        test_score_df = pd.read_csv(test_score_path)
        df_new = test_score_df[columns]
        df_all = pd.concat([df_all, df_new], axis=0,ignore_index=True)
    df_all['dataset'] = dataset_name
    df_all = df_all.reindex(columns=['dataset']+list(columns))
    df_all.to_csv(os.path.join(base_dir, dataset_name, 'metric_'+dataset_name+'.csv'), index=False, float_format='%.3f')
    print(df_all)

