import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report, matthews_corrcoef, cohen_kappa_score

def make_plot(args, predict_df, performance_df):
    #get data
    y_true = predict_df[args.target_name]
    y_pred = predict_df[args.pred_name]
    rmse = performance_df.loc[0, 'rmse']
    mse = performance_df.loc[0, 'mse']
    r2 = performance_df.loc[0, 'r2']
    mae = performance_df.loc[0, 'mae']

    #plot
    fontsize = 15
    fig, ax = plt.subplots(figsize=(8, 8))
    r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2), color="#5402A3")
    rmse_patch = mpatches.Patch(label="RMSE = {:.3f}".format(rmse), color="#5402A3")
    mse_patch = mpatches.Patch(label="MSE = {:.3f}".format(mse), color="#5402A3")
    mae_patch = mpatches.Patch(label="MAE = {:.3f}".format(mae), color="#5402A3")

    # r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2), color="darkcyan")
    # rmse_patch = mpatches.Patch(label="RMSE = {:.2f}".format(rmse), color="darkcyan")
    # mae_patch = mpatches.Patch(label="MAE = {:.2f}".format(mae), color="darkcyan")

    min_lim = math.floor(min(min(y_true), min(y_pred)))
    max_lim = math.ceil(max(max(y_true), max(y_pred)))
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.scatter(y_true, y_pred, alpha=0.2, color="#5402A3")
    # plt.scatter(y_true, y_pred, alpha=0.1, color="darkcyan")
    plt.plot(np.arange(min_lim, max_lim+1), np.arange(min_lim, max_lim+1), ls="--", c=".3")
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch, mse_patch], fontsize=fontsize)
    ax.set_ylabel('y_pred', fontsize=fontsize)
    ax.set_xlabel(args.target_name, fontsize=fontsize)
    plt.tick_params(labelsize=15, direction='inout', length=6, width=2, grid_alpha=0.5)
    # ax.set_title(model_name+'_'+checkpoint_dir.rsplit('/', 1)[-1], fontsize=20)
    ax.set_title(args.model_name + '_' + args.dataset_name, fontsize=20)
    plt.savefig(os.path.join(args.checkpoint_dir, args.model_name+'.jpg'))
    plt.show()
    return fig

def metric_regression(predict_df, args):
    # 决定系数（r2_score）
    r2 = round(r2_score(predict_df[args.target_name], y_pred=predict_df[args.pred_name]),3)
    # 平均绝对误差（mean absolute error）
    mae = round(mean_absolute_error(predict_df[args.target_name], y_pred=predict_df[args.pred_name]),3)
    # 均方差（mean squared error)
    mse = round(mean_squared_error(predict_df[args.target_name], y_pred=predict_df[args.pred_name]),3)
    # 平方根误差(root mean square error)
    rmse = round(math.sqrt(mse),3)
    performance_df = pd.DataFrame({'model_name': args.model_name,
                                   'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])
    performance_df.to_csv(os.path.join(args.checkpoint_dir, 'performance.csv'), index=False)
    return performance_df



def plot_confusion_matrix(args, confusion, class_labels, performance_df):
    fontsize = 15
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(confusion, cmap=plt.cm.Greens)
    indices = range(len(confusion))
    plt.xticks(indices, class_labels)
    plt.yticks(indices, class_labels)
    ax.set_title(args.model_name + '_' + args.dataset_name, fontsize=20)
    plt.colorbar()
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_true')
    # 显示数据
    for row, col in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        print(row, col, confusion[row, col])
        plt.text(col, row, confusion[row, col])
    # for first_index in range(len(confusion)):
    #     for second_index in range(len(confusion[first_index])):
    #         plt.text(first_index, second_index, confusion[first_index][second_index])

    #legend
    handles = []
    for idx, label in enumerate(class_labels):
        handles.append(mpatches.Patch(label=f"{label}_precision = {performance_df.loc[0, label + '_precision']:.3f}", color='green'))
    handles.append(mpatches.Patch(label=f"accuracy = {performance_df.loc[0, 'accuracy']:.3f}", color='green'))
    handles.append(mpatches.Patch(label=f"specificity = {performance_df.loc[0, 'specificity']:.3f}", color='green'))
    # handles.append(mpatches.Patch(label=f"recall = {performance_df.loc[0, 'recall']:.3f}", color='green'))
    # handles.append(mpatches.Patch(label="F1_score = {performance_df.loc[0,'f1']:.3f}, color='green'))
    # handles.append(mpatches.Patch(label=f"kappa = {performance_df.loc[0,'kappa']:.3f}", color='green'))
    plt.legend(handles=handles, fontsize=10)

    plt.savefig(os.path.join(args.checkpoint_dir, args.model_name + '_confusion.jpg'))
    plt.show()


def metric_classification(predict_df, args):
    y_true = predict_df[args.target_name]
    y_pred = predict_df['target_label_pred']
    predict_df['y_pred_int'] = predict_df['target_label_score'].apply(lambda x: 1 if x > 0.5 else 0)

    class_labels = sorted(pd.unique(predict_df[args.target_name]))
    performance_df = pd.DataFrame({'model': [args.model_name]})
    # 混淆矩阵
    confusion = confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    # classification_report ：综合评估
    report = classification_report(y_true, y_pred, digits=5, output_dict=True)
    for label in class_labels:
        performance_df.loc[0, label+'_precision'] = round(report[label]['precision'], 5)
    #准确率
    performance_df.loc[0,'accuracy'] = round(accuracy_score(y_true, y_pred), 5)
    #特异度specificity
    performance_df.loc[0,'specificity'] = round(TN / float(TN+FP), 5)
    # 召回率（灵敏度(Sensitivity)）
    Sensitivity = TP / float(TP + FN)
    performance_df.loc[0,'recall'] = round(recall_score(y_true, y_pred, pos_label=class_labels[-1]), 5)
    #精确率
    performance_df.loc[0, 'precision'] = round(precision_score(y_true, y_pred, pos_label=class_labels[-1]), 5)
    #F1-score
    performance_df.loc[0, 'f1'] = round(f1_score(y_true, y_pred, pos_label=class_labels[-1]), 5)
    # Kappa系数
    performance_df.loc[0, 'kappa'] = round(cohen_kappa_score(y_true, y_pred), 5)
    #马修斯相关系数（Matthews correlation coefficient）
    performance_df.loc[0, 'mcc'] = round(matthews_corrcoef(y_true, y_pred), 5)
    #AUC
    performance_df.loc[0, 'auc'] = round(roc_auc_score(predict_df['target_label'], predict_df['y_pred_int']), 5)

    plot_confusion_matrix(args, confusion, class_labels, performance_df)

    # save performance
    performance_df.to_csv(os.path.join(args.checkpoint_dir, 'performance.csv'), index=False)




def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--dataset_type', type=str, choices=['regression', 'classification', 'multiclass'],
                        default='classification', help='Type of dataset, e.g. classification or regression.'
                                                       'This determines the loss function used during training')
    parser.add_argument('--model_name', type=str, default='CMPNN', help='train model name')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset.csv file name')
    parser.add_argument('--target_name', type=str, required=True, help='data target column name')
    args = parser.parse_args()
    args.pred_name = args.target_name + '_pred'
    # print(args.data_path)
    # if not os.path.exists(os.path.dirname(args.save_path)):
    #     os.makedirs(os.path.dirname(args.save_path))
    # assert not os.path.exists(args.save_path), f'save_dir:{args.save_path} already exists'

    return args

if __name__ == '__main__':
    # checkpoint_dir = '/mnt/home/linjie/projects/CMPNN_original/ckpt/DMPNN/ELD_activity_value_lr0.001_fold5'
    # target_name ='activity_value'
    # checkpoint_dir = '/mnt/home/linjie/projects/CMPNN_original/ckpt_debug'
    # dataset_name = 'abc'
    # target_name = 'logSolubility'
    # pred_name = target_name + '_pred'
    # model_name = 'CMPNN'

    args = add_args()
    predict_df = pd.read_csv(os.path.join(args.checkpoint_dir, 'predict.csv'))


    #####metric#####
    if args.dataset_type == 'regression':
        performance_df = metric_regression(predict_df, args)
        make_plot(args=args, predict_df=predict_df, performance_df=performance_df)
    elif args.dataset_type == 'classification':
        performance_df = metric_classification(predict_df, args)






