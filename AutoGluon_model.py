import argparse
import itertools
import os
import pickle
import re
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import classification_report, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

def load_split_data(data_path):
    print(data_path)
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.1, random_state=0)
    train_data = TabularDataset(data=train)
    test_data = TabularDataset(data=test)
    print(f'train_data={len(train_data)}| test_data={len(test_data)}')
    return train_data, test_data

def train_model(args, train_data):
    print('train model')
    train_data.drop(columns=['smiles'], inplace=True)
    dataset_type = 'binary' if args.dataset_type == 'classification' else args.dataset_type
    label = args.target_columns if args.dataset_type == 'regression' else 'target_label'

    predictor = TabularPredictor(label=label, problem_type=dataset_type, path=args.save_dir, eval_metric=args.metric_type).fit(train_data, time_limit=args.time_limit, presets=args.presets)
    results = predictor.fit_summary(show_plot=True)
    print('------------------train model fiinshed------------------')
    # print(results)

    return predictor

def model_predict(args, test_data):
    predictor = TabularPredictor.load(args.save_dir)
    y_pred = predictor.predict(test_data.iloc[:, :1024])
    if args.dataset_type == 'regression':
        y_name = args.target_columns
        y_true = round(test_data.iloc[:, -1], 6)
        r2 = r2_score(y_true, y_pred)  # 决定系数（r2_score）
        mae = mean_absolute_error(y_true, y_pred)  # 平均绝对误差（mean absolute error）
        mse = mean_squared_error(y_true, y_pred)  # 均方差（mean squared error)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))  # 平方根误差(root mean square error)
        performance_df = pd.DataFrame({'model': predictor.get_model_best(),
                                       'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])
        plot_regression(args, performance_df, y_true, y_pred)

    else:
        y_name = 'target_label'
        average_type = 'binary' if args.dataset_type == 'classification' else 'micro'
        y_true = test_data.iloc[:, -1]
        with open(args.label_encoder_path, 'rb') as f:
            le = pickle.load(f)
        class_labels = le.classes_
        # y_true_str = le.inverse_transform(y_true)
        # y_pred_str = le.inverse_transform(y_pred)
        # 混淆矩阵
        confusion = confusion_matrix(y_true, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        performance_df = pd.DataFrame({'model': [predictor.get_model_best()]})
        performance_df.loc[0, 'kappa'] = round(cohen_kappa_score(y_true, y_pred), 3)
        performance_df.loc[0, 'accuracy'] = round(accuracy_score(y_true, y_pred), 3)
        report = classification_report(y_true, y_pred, output_dict=True, digits=3)
        for idx, label in enumerate(class_labels):
            performance_df.loc[0, label+'_precision'] = report[str(idx)]['precision']
        performance_df.loc[0, 'mcc'] = round(matthews_corrcoef(y_true, y_pred), 3)
        performance_df.loc[0, 'recall'] = round(recall_score(y_true, y_pred, average=average_type), 3)
        performance_df.loc[0, 'f1'] = round(f1_score(y_true, y_pred, average=average_type), 3)
        # performance_df.loc[0, 'precision'] = round(precision_score(y_true, y_pred, average=average_type), 3)
        if args.dataset_type != 'multiclass':
            performance_df.loc[0, 'auc'] = round(roc_auc_score(y_true, y_pred), 3)
            performance_df.loc[0, 'specificity'] = round(TN / float(TN + FP), 3)

        plot_confusion_matrix(args, confusion, class_labels, performance_df)

    print(performance_df)
    performance_df.to_csv(os.path.join(args.save_dir, 'performance.csv'), index=False, float_format='%.3f')
    df_pred = pd.DataFrame({'smiles': test_data['smiles'], y_name: y_true, y_name + '_pred': y_pred})
    df_pred.to_csv(os.path.join(args.save_dir, 'predict.csv'), index=False)



def plot_regression(args, performance_df, y_true, y_pred):
    fontsize = 15
    fig, ax = plt.subplots(figsize=(8, 8))
    # legend
    handles = []
    handles.append(mpatches.Patch(label=f"R2 = {performance_df.loc[0, 'r2']:.3f}", color="#5402A3"))
    handles.append(mpatches.Patch(label=f"RMSE = {performance_df.loc[0, 'rmse']:.3f}", color="#5402A3"))
    handles.append(mpatches.Patch(label=f"MAE = {performance_df.loc[0, 'mae']:.3f}", color="#5402A3"))
    handles.append(mpatches.Patch(label=f"MSE = {performance_df.loc[0, 'mse']:.3f}", color="#5402A3"))

    min_lim = math.floor(min(min(y_true), min(y_pred)))
    max_lim = math.ceil(max(max(y_true), max(y_pred)))
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.scatter(y_true, y_pred, alpha=0.2, color="#5402A3")
    plt.plot(np.arange(min_lim, max_lim+1), np.arange(min_lim, max_lim+1), ls="--", c=".3")
    plt.legend(handles=handles, fontsize=fontsize)
    ax.set_ylabel('y_pred', fontsize=fontsize)
    ax.set_xlabel(args.target_columns, fontsize=fontsize)
    plt.tick_params(labelsize=15, direction='inout', length=6, width=2, grid_alpha=0.5)
    ax.set_title(performance_df.loc[0, 'model'] + '_' + args.dataset_name, fontsize=20)
    plt.savefig(os.path.join(args.save_dir, performance_df.loc[0,'model'] + '.jpg'))
    # plt.show()
    return fig

def plot_confusion_matrix(args, confusion, class_labels, performance_df):
    fontsize = 15
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(confusion, cmap=plt.cm.Greens)
    indices = range(len(confusion))
    plt.xticks(indices, class_labels)
    plt.yticks(indices, class_labels)
    plt.colorbar()
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_true')
    ax.set_title(performance_df.loc[0, 'model'] + '_' + args.dataset_name, fontsize=20)
    # 显示数据
    for row, col in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        # print(row, col, confusion[row, col])
        plt.text(col, row, confusion[row, col])

    #legend
    handles = []
    # for idx, label in enumerate(class_labels):
    #     handles.append(mpatches.Patch(label=f"{label}_precision = {performance_df.loc[0, label + '_precision']:.3f}", color='green'))
    handles.append(mpatches.Patch(label=f"kappa = {performance_df.loc[0,'kappa']:.3f}", color='green'))
    handles.append(mpatches.Patch(label=f"accuracy = {performance_df.loc[0, 'accuracy']:.3f}", color='green'))
    handles.append(mpatches.Patch(label=f"recall = {performance_df.loc[0, 'recall']:.3f}", color='green'))
    handles.append(mpatches.Patch(label=f"F1_score = {performance_df.loc[0,'f1']:.3f}", color='green'))
    # handles.append(mpatches.Patch(label=f"specificity = {performance_df.loc[0, 'specificity']:.3f}", color='green'))


    plt.legend(handles=handles, fontsize=8)

    plt.savefig(os.path.join(args.save_dir, performance_df.loc[0,'model'] + '_confusion.jpg'))
    # plt.show()


def add_args():
    metric_list = ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
                   'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro',
                   'precision_micro',
                   'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss',
                   'pac_score',
                   'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                   'mean_absolute_percentage_error', 'r2', 'kappa', 'quadratic_kappa']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='original data path')
    parser.add_argument('--save_dir', type=str, required=True, help='specifies folder to store trained models')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset file name')
    parser.add_argument('--dataset_type', type=str, default='regression',
                        choices=['classification', 'multiclass', 'regression', 'quantile'],
                        help='type of prediction problem this Predictor has been trained for')
    parser.add_argument('--target_columns', type=str, help='Name of the columns containing target values.')
    parser.add_argument('--metric_type', type=str, default=None,
                        choices=metric_list,
                        help='metric is used to evaluate predictive performance')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help='how long model train run (wallclock time in seconds).')
    parser.add_argument('--presets', type=str, default='best_quality',
                        choices=['best_quality', 'high_quality', 'good_quality', 'medium_quality',
                                 'optimize_for_deployment', 'interpretable', 'ignore_text'],
                        help='Can significantly impact predictive accuracy')
    parser.add_argument('--class_labels', type=str, nargs='+', default=None,
                        help='For multiclass problems, this list contains the class labels in sorted order of `predict_proba()` output.')
    parser.add_argument('--positive_class', type=str, default=None,
                        help='Returns the positive class name in binary classification.')
    parser.add_argument('--testing', action="store_true", default=False,
                        help='predict instead of training model')

    args = parser.parse_args()
    if not args.testing:
        assert not os.path.exists(args.save_dir), f'save_dir:{args.save_dir} already exists'
        os.makedirs(args.save_dir)
    # # if os.path.exists(save_path):
    # #     shutil.rmtree(save_path)

    if args.metric_type == None:
        if args.dataset_type != 'regression':
            args.metric_type = 'quadratic_kappa'
        else:
            args.metric_type = 'root_mean_squared_error'
    if args.dataset_type != 'regression':
        args.label_encoder_path = args.data_path.replace('.csv', '.pkl').replace('ECFP_', '')
    else:
        args.label_encoder_path = None
    # args.dataset_name = Path(args.label_encoder_path).stem

    """
    python AutoGluon_model.py \
    --data_path /mnt/home/linjie/projects/solubility/data/regression/Gostar_LogD_Solubility/ESL_pLogS/ECFP_ESL_pLogS.csv \
    --save_dir /mnt/home/linjie/projects/solubility/AutogluonModels/debug \
    --dataset_name ESL_pLogS \
    --dataset_type regression \
    --target_columns pLogS \
    --metric_type r2 \
    --presets medium_quality


    python AutoGluon_model.py \
    --data_path /mnt/home/linjie/projects/solubility/data/regression/Gostar_LogD_Solubility/ELD_activity_value/ECFP_ELD_activity_value.csv \
    --save_dir AutogluonModels/ELD_activity_value \
    --dataset_name ELD_activity_value \
    --dataset_type regression \
    --target_columns activity_value \
    --metric_type r2

    python AutoGluon_model.py \
    --data_path /mnt/home/linjie/projects/solubility/data/classification/modelling_data_1/HHep_T_half/ECFP_HHep_T_half.csv \
    --save_dir /mnt/home/linjie/projects/solubility/AutogluonModels/debug \
    --dataset_name target_label \
    --dataset_type multiclass \
    --metric_type quadratic_kappa \
    --presets medium_quality


    --data_path /mnt/home/linjie/projects/solubility/data/classification/CYP_hERG_clean_data/CYP1A2_data_clean/ECFP_CYP1A2_data_clean.csv \
    --save_dir /mnt/home/linjie/projects/solubility/AutogluonModels/debug \
    --dataset_name target_label \
    --dataset_type classification \
    --metric_type quadratic_kappa \
    --presets medium_quality

    """

    return args

if __name__ == '__main__':
    args = add_args()
    train_data, test_data = load_split_data(args.data_path)
    if not args.testing:
        predictor = train_model(args, train_data)
    model_predict(args, test_data)





"""
python /mnt/home/linjie/projects/solubility/AutoGluon_model.py \
--data_path /mnt/home/linjie/projects/solubility/data/regression/Gostar_LogD_Solubility/ESL_pLogS/ECFP_ESL_pLogS.csv \
--save_dir /mnt/home/linjie/projects/solubility/result/regression/Gostar_LogD_Solubility/ESL_pLogS/AutoGluonModel \
--dataset_name ESL_pLogS \
--dataset_type regression \
--target_columns pLogS \
--metric_type r2 \
--presets medium_quality \
--testing
"""












# from autogluon.tabular import TabularDataset, TabularPredictor
# train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
# subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
# train_data = train_data.sample(n=subsample_size, random_state=0)
# train_data.head()
# label = 'class'
# print("Summary of class variable: \n", train_data[label].describe())
# save_path = 'agModels-predictClass'  # specifies folder to store trained models
# predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
# test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
# y_test = test_data[label]  # values to predict
# test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
# test_data_nolab.head()
# predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file
# y_pred = predictor.predict(test_data_nolab)
# print("Predictions:  \n", y_pred)
# perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
# predictor.leaderboard(test_data, silent=True)


# cls_report = list(filter(None, re.split(r'\n', cls_report.strip())))
# cls_report_df = pd.DataFrame(columns=['label', 'precision', 'recall', 'f1-score', 'support'])
# for idx, value in enumerate(cls_report[1:]):
#     print(idx)
#     cls_report_df.loc[idx, :] = re.split(r'\s+', value.strip())
#     if idx == 3:
#         l = re.split(r'\s+', value.strip())
#         cls_report_df.loc[idx, :] = [l[0], 'NAN', 'NAN', l[1], l[2]]
#
#     elif idx in [4, 5]:
#         l = re.split(r'\s+', value.strip())
#         l[0] = l[0]+'_'+l[1]
#         l.remove('avg')
#         cls_report_df.loc[idx, :] = l
#     cls_report_df.to_csv(os.path.join(save_dir, 'performance.csv'), index=False)


    # #access various information about the trained predictor or a particular model
    # all_models = predictor.get_model_names()
    # model_to_use = all_models[19]
    # specific_model = predictor._trainer.load_model(model_to_use)
    # # Objects defined below are dicts of various information (not printed here as they are quite large):
    # model_info = specific_model.get_info()
    # predictor_information = predictor.info()
