import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from tqdm import tqdm


# from myLabelEncoder import MyLabelEncoder




def smiles_to_ecfp(smiles, size=1024):
    """Converts a single SMILES into an ECFP4

    Parameters:
        smiles (str): The SMILES string .
        size (int): Size (dimensions) of the ECFP4 vector to be calculated.

    Returns:
        ecfp4 (arr): An n dimensional ECFP4 vector of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    arr = np.zeros((0,), dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(ecfp, arr)
    return arr


def get_df_ecfp(df, target_name, dataset_type):
    print('get_df_ecfp')
    arr_list = []
    for idx, el in df.iterrows():
        arr = smiles_to_ecfp(el['smiles'])
        arr_list.append(arr)
    new_df = pd.DataFrame(arr_list)
    new_df['smiles'] = df['smiles']
    new_df[target_name] = df[target_name]
    if dataset_type != 'regression':
        new_df['target_label'] = df['target_label']

    return new_df


def remove_salt(data_path, target_name):
    df = pd.read_excel(data_path)
    df.dropna(subset=[target_name, 'sub_smiles'], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    for idx, smiles in tqdm(enumerate(df['sub_smiles']), total=len(df)):
        smiles_list = smiles.split('.')
        smi = max(smiles_list, key=len, default='')
        mol = Chem.MolFromSmiles(smi)
        if mol:
            df.loc[idx, 'smiles'] = Chem.MolToSmiles(mol, canonical=True)
    df.dropna(subset=['smiles'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def df_drop_duplicated(df, target_name, dataset_type, save_path):
    base_dir = os.path.join(os.path.abspath(os.path.join(save_path, "../..")))
    base_dir_name = Path(base_dir).stem

    data_summary = {base_dir_name: Path(save_path).stem, 'raw': len(df)}
    print('before len(df)=', len(df))
    df_duplicated = df.loc[df.duplicated(subset='smiles', keep=False), :]
    smiles_key = list(set(df_duplicated['smiles']))
    print(f'df_duplicated: {len(df_duplicated)}| smiles_key:{len(smiles_key)}')
    if dataset_type == 'regression':
        # fun = lambda x: x.fill(df.groupby('smiles').target_name.mean()[x.target_name])
        # df = df.apply(lambda x: fun(x), axis=1)
        for key in smiles_key:
            target_value = np.mean(df.loc[df['smiles'] == key, target_name])
            df.loc[df['smiles'] == key, target_name] = target_value
        df.drop_duplicates(subset='smiles', keep='first', inplace=True, ignore_index=True)
        data_summary.update({'processed': len(df), 'target_name': target_name})
    else:
        invalid_target_value = []
        print(f'data[{target_name}] classification level: {pd.unique(df[target_name]).tolist()}')
        df_groups = df.groupby('smiles')
        for smiles, group in df_groups:
            if len(set(group[target_name])) > 1:
                invalid_target_value.append(smiles)
        df = df.loc[~df['smiles'].isin(invalid_target_value), :]
        df.reset_index(drop=True, inplace=True)
        df = df.drop_duplicates(subset='smiles', keep='first', ignore_index=True)
        data_summary.update({'processed': len(df), 'target_name': target_name, 'level': str(pd.unique(df[target_name]))})
    print('after len(df)=', len(df))
    print('after df_duplicated', len(df.loc[df.duplicated(subset='smiles', keep=False), :]))

    df.to_csv(save_path, index=False)
    if os.path.exists(os.path.join(base_dir, base_dir_name+'.csv')):
        header=False
    else:
        header=True
    pd.DataFrame(data_summary, index=[0]).to_csv(os.path.join(base_dir, base_dir_name+'.csv'), mode='a', header=header, index=False)

    return df

# def label_encoder(df, target_name, label_encoder_path):
#     le = LabelEncoder()
#     df['target_label'] = le.fit_transform(df[target_name])
#     with open(label_encoder_path, 'wb') as f:
#         pickle.dump(le, f)
#
#
#     # with open(label_encoder_path, 'rb') as f:
#     #     le_departure = pickle.load(f)
#     #     df['target_label'] = le.transform(df[target_name])
#     return df

class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self



def label_encoder(df, target_name, label_encoder_path, dataset_type):
    label = np.unique(df[target_name])
    if 'High' in label:
        label = label[::-1]
    else:
        label = label
    le = LabelEncoder()
    le.fit(label)
    le.classes_ = pd.Series(label).unique()
    df['target_label'] = le.transform(df[target_name])
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(le, f)

    # with open(label_encoder_path, 'rb') as f:
    #     le_departure = pickle.load(f)
    #     df['target_label'] = le.transform(df[target_name])
    return df



def classification_subplot(ax, data, save_path, target_name, data_type='train'):
    target_value = data.groupby(target_name).size().to_dict()
    target_level = list(target_value.keys())
    ax.bar(target_level, list(target_value.values()), width=0.3, color=[0.012, 0.5, 0.5], edgecolor='k')
    ax.set_title(data_type+'_' + Path(save_path).stem)
    ax.set_xlabel(target_name)
    ax.set_ylabel('count')

def regression_subplot(ax, data, save_path, target_name, data_type='train'):
    ax.hist(data[target_name], bins=20, color=[0.8, 0.5, 0.5], edgecolor='k')
    ax.set_title(data_type+'_' + Path(save_path).stem)
    ax.set_xlabel(target_name)
    ax.set_ylabel('count')


def plot_data(train, test, save_path, target_name, dataset_type):
    all_data = pd.concat([train, test], axis=0)
    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.style.use('ggplot')
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    if dataset_type == 'regression':
        regression_subplot(fig.add_subplot(1, 2, 1), train, save_path, target_name)
        regression_subplot(fig.add_subplot(1, 2, 2), test, save_path, target_name)
        fig.savefig(save_path)
        # fig.show()
        fig.clf()
        regression_subplot(fig.add_subplot(1, 1, 1), all_data, save_path.replace('.jpg', '_all_data.jpg'), target_name, '')
        fig.savefig(save_path.replace('.jpg', '_all_data.jpg'))
        fig.show()
    else:
        classification_subplot(fig.add_subplot(1, 2, 1), train, save_path, target_name, 'train')
        classification_subplot(fig.add_subplot(1, 2, 2), test, save_path, target_name, 'test')
        fig.savefig(save_path)
        # fig.show()
        fig.clf()
        classification_subplot(fig.add_subplot(1, 1, 1), all_data, save_path.replace('.jpg', '_all_data.jpg'), target_name, '')
        fig.savefig(save_path.replace('.jpg', '_all_data.jpg'))
        fig.show()




def plot_regression(data, save_path, target_name):
    plt.style.use('ggplot')
    plt.hist(data[target_name], bins=20, color=[0.8, 0.5, 0.5], edgecolor='k')
    plt.title(Path(save_path).stem)
    plt.xlabel(target_name)
    plt.ylabel('count')
    plt.show()
    plt.savefig(os.path.join(save_path))


def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to raw data xlsx file')
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'regression', 'milticlass'],
                        default='classification', help='Type of dataset, e.g. classification or regression.'
                                                       'This determines the loss function used during training')
    parser.add_argument('--save_path', type=str, help='save .csv path of data after processing')
    parser.add_argument('--target_name', type=str, help='data target column name')
    args = parser.parse_args()
    # print(args.data_path)
    # if not os.path.exists(os.path.dirname(args.save_path)):
    #     os.makedirs(os.path.dirname(args.save_path))
    if args.save_path is not None:
        assert not os.path.exists(args.save_path), f'save_dir:{args.save_path} already exists'

    return args


if __name__ == '__main__':
    """
    --data_path ./data/raw/regression/ELD_activity_value.xlsx
    --save_path ./data/regression/ELD_activity_value/ELD_activity_value.csv
    --target_name activity_value
    --dataset_type regression
    
    
    
    --data_path ./data/raw/regression/ESL_pLogS.xlsx
    --save_path ./data/regression/ESL_pLogS/ESL_pLogS.csv
    --target_name pLogS
    --dataset_type regression
    
    --data_path ./data/raw/classification/HHep_T_half.xlsx
    --save_path ./data/classification/HHep_T_half/HHep_T_half.csv
    --target_name t1/2_Class
    --dataset_type classification
    
    classification={'F_data': 'LogF_Class', 'HHep_T_half': 't1/2_Class', 'HHepClint': 'HHepClint_Class', 'HLM_T_half': 't1/2_Class',
     'HLMClint_ml_min_kg': 'HLM_Class', 'HLMClint_ul_min_mg': 'HLM_Class','MDCK_ER': 'ER_Class', 'MHep_T_half': 't1/2_Class',
     'MLM_T_half': 't1/2_Class', 'PAMPA_permeability': 'CNS_Perm_Class', 'RHep_T_half': 't1/2_Class', 'RHepClint': 'RHepClint_Class',
     'RLM_T_half': 't1/2_Class', 'RLMClint_ml_min_kg': 'RLM_Class', 'RLMClint_ul_min_mg': 'RLM_Class'}
    regression={'ELD_activity_value': 'activity_value', 'ESL_pLogS': 'pLogS'}
    """

    modelling_data_1 = {'F_data': 'LogF_Class', 'HHep_T_half': 't1/2_Class', 'HHepClint': 'HHepClint_Class',
                        'HLM_T_half': 't1/2_Class', 'HLMClint_ml_min_kg': 'HLM_Class',
                        'HLMClint_ul_min_mg': 'HLM_Class', 'MDCK_ER': 'ER_Class', 'MHep_T_half': 't1/2_Class',
                        'MLM_T_half': 't1/2_Class', 'PAMPA_permeability': 'CNS_Perm_Class', 'RHep_T_half': 't1/2_Class',
                        'RHepClint': 'RHepClint_Class', 'RLM_T_half': 't1/2_Class', 'RLMClint_ml_min_kg': 'RLM_Class',
                        'RLMClint_ul_min_mg': 'RLM_Class'}
    Gostar_LogD_Solubility = {'ELD_activity_value': 'activity_value', 'ESL_pLogS': 'pLogS'}
    CYP_hERG_clean_data = {'CYP1A2_data_clean': 'CYP1A2_Class', 'CYP2C9_data_clean': 'CYP2C9_Class',
                           'CYP2C19_data_clean': 'CYP2C19_Class', 'CYP2D6_data_clean': 'CYP2D6_Class',
                           'CYP3A4_data_clean': 'CYP3A4_Class', 'HERG-220720_clean': 'hERG_Class'}
    for dataname, datatype in {'CYP_hERG_clean_data': 'classification'}.items():
        for key, value in CYP_hERG_clean_data.items():
            args = add_args()
            args.dataset_type = datatype
            args.data_path = f'./data/raw/{args.dataset_type}/{dataname}/{key}.xlsx'
            args.save_path = f'./data/{args.dataset_type}/{dataname}/{key}/{key}.csv'

            args.target_name = value
            print(args.data_path)
            if not os.path.exists(os.path.dirname(args.save_path)):
                os.makedirs(os.path.dirname(args.save_path))
            assert not os.path.exists(args.save_path), f'save_dir:{args.save_path} already exists'
            # raw data processed
            df = remove_salt(args.data_path, args.target_name)
            df = df_drop_duplicated(df, args.target_name, args.dataset_type, args.save_path)


            # data only retains the [smiles, target] column
            df_smiles_target = df.loc[:, ['smiles', args.target_name]]
            if args.dataset_type != 'regression':
                df_smiles_target = label_encoder(df_smiles_target, args.target_name, args.save_path.replace('csv', 'pkl'), args.dataset_type)
            df_smiles_target.to_csv('/smiles_target_'.join(args.save_path.rsplit('/', 1)), index=False)

            # get ecfp
            # if args.dataset_type != 'regression':
            #     target_name = 'target_label'
            # else:
            #     target_name = args.target_name
            ecfp_df = get_df_ecfp(df_smiles_target, args.target_name, args.dataset_type)
            ecfp_df.to_csv('/ECFP_'.join(args.save_path.rsplit('/', 1)), index=False)

            #split data
            train, test = train_test_split(df_smiles_target, test_size=0.1, random_state=0)
            train.to_csv('/train_'.join(args.save_path.rsplit('/', 1)), index=False)
            test.to_csv('/test_'.join(args.save_path.rsplit('/', 1)), index=False)
            plot_data(train, test, args.save_path.replace('csv', 'jpg'), args.target_name, args.dataset_type)
            print('*' * 100)
