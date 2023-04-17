import argparse
import os
import re
from math import sqrt
import torch
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, average_precision_score, \
    confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, precision_score
from sklearn.model_selection import train_test_split


def add_args():
    metric_list = ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
                   'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro',
                   'precision_micro',
                   'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss',
                   'pac_score',
                   'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                   'mean_absolute_percentage_error', 'r2']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='original data path')
    parser.add_argument('--save_dir', type=str, required=True, help='specifies folder to store trained models')
    parser.add_argument('--dataset_type', type=str, default='regression',
                        choices=['binary', 'multiclass', 'regression', 'quantile'],
                        help='type of prediction problem this Predictor has been trained for')
    parser.add_argument('--metric_type', type=str, default=None,
                        choices=metric_list,
                        help='metric is used to evaluate predictive performance')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help=' how long model train run (wallclock time in seconds).')
    parser.add_argument('--presets', type=str, default='best_quality',
                        choices=['best_quality', 'high_quality', 'good_quality', 'medium_quality',
                                 'optimize_for_deployment', 'interpretable', 'ignore_text'],
                        help='Can significantly impact predictive accuracy')
    parser.add_argument('--class_labels', type=str, nargs='+', default=None,
                        help='For multiclass problems, this list contains the class labels in sorted order of `predict_proba()` output.')
    parser.add_argument('--positive_class', type=str, default=None,
                        help='Returns the positive class name in binary classification.')

    args = parser.parse_args()
    assert not os.path.exists(args.save_dir), f'save_dir:{args.save_dir} already exists'
    os.makedirs(args.save_dir)
    # # if os.path.exists(save_path):
    # #     shutil.rmtree(save_path)

    if args.metric_type == None:
        if args.dataset_type == 'classification':
            args.metric_type = 'accuracy'
        else:
            args.metric_type = 'root_mean_squared_error'

    """
    python AutoGluon_model.py \
    --data_path /mnt/home/linjie/projects/solubility/data/regression/ESL_pLogS/ECFP_ESL_pLogS.csv \
    --save_dir /mnt/home/linjie/projects/solubility/AutogluonModels/ESL_pLogS \
    --dataset_type regression \
    --metric_type r2


    --presets medium_quality


    python AutoGluon_model.py \
    --data_path /mnt/home/linjie/projects/solubility/data/regression/ELD_activity_value/ECFP_ELD_activity_value.csv \
    --save_dir AutogluonModels/ELD_activity_value \
    --dataset_type regression \
    --metric_type r2

    python AutoGluon_model.py \
    --data_path /mnt/home/linjie/projects/solubility/data/classification/modelling_data_1/HHep_T_half/ECFP_HHep_T_half.csv \
    --save_dir AutogluonModels/classification/modelling_data_1/HHep_T_half \
    --dataset_type multiclass \
    --metric_type accuracy \
    --class_labels High Medium Low
    --presets medium_quality
    """

    return args


def load_split_data(data_path):
    print(data_path)
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.1, random_state=0)
    train_data = TabularDataset(data=train)
    test_data = TabularDataset(data=test)
    print(f'train_data={len(train_data)}| test_data={len(test_data)}')
    return train_data, test_data


def train_model(args):
    print('train model')
    label = 'y'
    predictor = TabularPredictor(label=label, problem_type=args.dataset_type, path=args.save_dir,
                                 eval_metric=args.metric_type).fit(train_data, time_limit=args.time_limit,
                                                                   presets=args.presets)
    results = predictor.fit_summary(show_plot=True)
    print('------------------train model fiinshed------------------')
    # print(results)

    return predictor


def model_predict(save_dir, dataset_type):
    predictor = TabularPredictor.load(save_dir)
    y_pred = predictor.predict(test_data.iloc[:, :-1])
    # perf = predictor.evaluate_predictions(y_true=test_data.iloc[:, -1], y_pred=y_pred, auxiliary_metrics=True)
    # performance = predictor.evaluate(test_data)
    # evaluate the performance of each individual trained model
    # print(predictor.leaderboard(test_data, silent=True))
    # save predict.csv
    if dataset_type == 'regression':
        y_true = round(test_data.iloc[:, -1], 6)
        r2 = r2_score(y_true, y_pred)  # 决定系数（r2_score）
        mae = mean_absolute_error(y_true, y_pred)  # 平均绝对误差（mean absolute error）
        mse = mean_squared_error(y_true, y_pred)  # 均方差（mean squared error)
        rmse = sqrt(mean_squared_error(y_true, y_pred))  # 平方根误差(root mean square error)
        performance_df = pd.DataFrame({'model_name': predictor.get_model_best(),
                                       'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])
        print(performance_df)
        performance_df.to_csv(os.path.join(save_dir, 'performance.csv'), index=False)


    elif dataset_type == 'multiclass':
        y_true = test_data.iloc[:, -1]

        acc = accuracy_score(y_true, y_pred)  # 准确率
        confusion = confusion_matrix(y_true=y_true, y_pred=y_pred)  # 混淆矩阵
        cls_report = classification_report(y_true=y_true, y_pred=y_pred,
                                           output_dict=True)  # precision, recall, f1_score, support
        performance_df = pd.DataFrame(cls_report).T

        perf = predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred, auxiliary_metrics=True)
        perf['model_name'] = predictor.get_model_best()
        performance_df = pd.DataFrame(perf, index=[0])

    df_pred = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df_pred.to_csv(os.path.join(save_dir, 'predict.csv'), index=False)



if __name__ == '__main__':
    args = add_args()
    train_data, test_data = load_split_data(args.data_path)
    predictor = train_model(args)
    model_predict(args.save_dir, args.dataset_type)

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
