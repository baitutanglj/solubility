import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def plot_target_value(data, title, target_name, target_level):
    # plt.figure(figsize=(5,3))
    plt.style.use('ggplot')
    plt.hist(data[target_name], bins=len(target_level), rwidth=0.3, color=[0.01766472354707649, 0.5, 0.5], edgecolor='k')
    plt.title(title)
    # plt.xticks([0.3,0.7], target_level)
    plt.xlabel(target_name)
    plt.ylabel('count')
    plt.show()


def plot_classification(data, save_path, target_name):
    plt.style.use('ggplot')
    target_value = data.groupby(target_name).size().to_dict()
    target_level = list(target_value.keys())
    plt.bar(target_level, list(target_value.values()),  width=0.3, color=[0.012, 0.5, 0.5], edgecolor='k')
    plt.title(Path(save_path).stem)
    # plt.xticks([index-0.8 for index in range(len(target_level))], target_level)
    plt.xlabel(target_name)
    plt.ylabel('count')
    plt.show()
    plt.savefig(os.path.join(save_path), dpi=300)



def add_args():
    metric_list = ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
                   'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro', 'precision_micro',
                   'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score',
                   'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                   'mean_absolute_percentage_error', 'r2']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='original data path')
    parser.add_argument('--save_dir', type=str, required=True, help='specifies folder to store trained models')
    parser.add_argument('--model_type', type=str, default='regression', choices=['regression', 'classification'],
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


    args = parser.parse_args()
    assert not os.path.exists(args.save_dir), f'save_dir:{args.save_dir} already exists'
    os.makedirs(args.save_dir)
    # # if os.path.exists(save_path):
    # #     shutil.rmtree(save_path)

    if args.metric_type == None:
        if args.model_type == 'classification':
            args.metric_type = 'accuracy'
        else:
            args.metric_type = 'root_mean_squared_error'


data = pd.read_csv("/mnt/home/linjie/projects/solubility/data/classification/F_data/train_F_data.csv")
title='train_F_data.csv'
target_name='LogF_Class'
target_level = pd.unique(data[target_name])
plot_target_value(data, title, target_name, target_level)


if __name__ == '__main__':
    target_name = 'LogF_Class'
    save_path = "~/Downloads/color.png"
    data = pd.read_csv("/mnt/home/linjie/projects/solubility/data/classification/F_data/train_F_data.csv")[target_name]
    # 创建x轴坐标
    data_class = pd.unique(data).tolist()
    # 绘制彩色条形图并保存
    plot_bar_color(data, data_class, target_name, save_path)
    # 绘制普通条形图并保存
    plot_bar_normal(data, x)
    print("All Done!")
