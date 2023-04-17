import os
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def make_plot(base_dir, x_label, y_true, y_pred, rmse, r2, mae, model_name):
    fontsize = 15
    fig, ax = plt.subplots(figsize=(8, 8))
    r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2), color="#5402A3")
    rmse_patch = mpatches.Patch(label="RMSE = {:.3f}".format(rmse), color="#5402A3")
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
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=fontsize)
    ax.set_ylabel('y_pred', fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    plt.tick_params(labelsize=15, direction='inout', length=6, width=2, grid_alpha=0.5)
    ax.set_title(base_dir.rsplit('/', 1)[-1]+'_'+model_name, fontsize=20)
    plt.savefig(os.path.join(base_dir, model_name+'.png'), dpi=300)
    plt.show()
    return fig



if __name__ == '__main__':
    base_dir = './AutogluonModels/ELD_activity_value'
    x_label ='activity_value'
    # base_dir = './AutogluonModels/ESL_pLogS'
    # x_label = 'pLogS'
    predict_df = pd.read_csv(os.path.join(base_dir, 'predict.csv'))
    performance_df = pd.read_csv(os.path.join(base_dir, 'performance.csv'))
    make_plot(base_dir=base_dir, x_label=x_label,
              y_true=predict_df['y_true'], y_pred=predict_df['y_pred'],
              rmse=performance_df.loc[0, 'rmse'], r2=performance_df.loc[0, 'r2'],
              mae=performance_df.loc[0, 'mae'], model_name=performance_df.loc[0, 'model_name'])



