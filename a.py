import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

iris = datasets.load_iris()
n_class = len(set(iris.target))  # 类别数量

x, y = iris.data, iris.target
y_one_hot = label_binarize(y, classes=np.arange(n_class))  # 转化为one-hot
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

X_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

# 建模
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(x_train_std, y_train)
model.predict_proba(x_test_std[:2, :])  # 查看前两个测试样本属于各个类别的概率
# 预测test结果
y_pred = model.predict(x_test_std)
# 预测train结果
y_pred = model.predict(x_train_std)

con_matrix = confusion_matrix(y_test, y_pred)
print('confusion_matrix:\n', con_matrix)
print('accuracy:{}'.format(accuracy_score(y_test, y_pred)))
print('precision micro:{}'.format(precision_score(y_test, y_pred, average='micro')))
print('recall micro:{}'.format(recall_score(y_test, y_pred, average='micro')))
print('f1-score micro:{}'.format(f1_score(y_test, y_pred, average='micro')))
# AUC值
y_one_hot = label_binarize(y_test, classes=np.arange(n_class))  # 转化为one-hot
y_pre_pro = model.predict_proba(x_test_std)
auc = roc_auc_score(y_one_hot, y_pre_pro, average='micro')
# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_pre_pro.ravel())
plt.plot(fpr, tpr, linewidth=2, label='AUC=%.3f' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1.1, 0, 1.1])
plt.xlabel('False Postivie Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
#classification_report ：综合评估，是评判模型便捷且全面的方法（参数digits控制精度）
report = classification_report(y_test,y_pred,digits=5)

print('precision macro:{}'.format(precision_score(y_test, y_pred, average='macro')))
print('recall macro:{}'.format(recall_score(y_test, y_pred, average='macro')))
print('f1-score macro:{}'.format(f1_score(y_test, y_pred, average='macro')))
auc = roc_auc_score(y_one_hot, y_pre_pro, average='macro')
y_one_hot = label_binarize(y_test, classes=np.arange(n_class))  # 转化为one-hot
y_pre_pro = model.predict_proba(x_test_std)
auc = roc_auc_score(y_one_hot, y_pre_pro, average='micro')




#########################
# 预测train结果
y_pred = model.predict(x_train_std)
con_matrix = confusion_matrix(y_train, y_pred)
print('confusion_matrix:\n', con_matrix)
print('accuracy:{}'.format(accuracy_score(y_train, y_pred)))
print('precision micro:{}'.format(precision_score(y_train, y_pred, average='micro')))
print('recall micro:{}'.format(recall_score(y_train, y_pred, average='micro')))
print('f1-score micro:{}'.format(f1_score(y_train, y_pred, average='micro')))
# AUC值
y_one_hot = label_binarize(y_train, classes=np.arange(n_class))  # 转化为one-hot
y_pre_pro = model.predict_proba(x_train_std)
print('auc micro',roc_auc_score(y_one_hot, y_pre_pro, average='micro'))
# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_pre_pro.ravel())
plt.plot(fpr, tpr, linewidth=2, label='AUC=%.3f' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1.1, 0, 1.1])
plt.xlabel('False Postivie Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
#classification_report ：综合评估，是评判模型便捷且全面的方法（参数digits控制精度）
print('classification_report:\n', classification_report(y_train, y_pred, digits=5))


print('pred:\n', confusion_matrix(y_true, y_pred, normalize='pred'))
print('true:\n', confusion_matrix(y_true, y_pred, normalize='true'))
print('all:\n', confusion_matrix(y_true, y_pred, normalize='all'))




def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)



import pandas as pd
score = pd.read_csv('/mnt/home/linjie/projects/solubility/result/classification/CYP_hERG_clean_data/CYP1A2_data_clean/DMPNN/test_scores.csv')
column_manes = score.columns
score.loc[:, ['Mean' in x for x in column_manes]]


# data_path = '/mnt/home/linjie/projects/solubility/AutogluonModels/classification/modelling_data_1/HLMClint_ul_min_mg/predict.csv'
data_path = '/mnt/home/linjie/projects/solubility/AutogluonModels/classification/modelling_data_1/F_data/predict.csv'
df = pd.read_csv(data_path)
a = df['y_true']=='POS'
b = df['y_pred']=='POS'
sum(a&b)

import os
from glob import glob
"""
['HHep_T_half', 'HLMClint_ul_min_mg', 'RLMClint_ul_min_mg', 'RHepClint', 'RLMClint_ml_min_kg', 
'MLM_T_half', 'F_data', '{MHep_T_half}', 'HHepClint', 'HLMClint_ml_min_kg', '{RHep_T_half}', 'HLM_T_half', 'PAMPA_permeability', 'RLM_T_half', 'MDCK_ER']

"""
"""
MHep_T_half, HLMClint_ul_min_mg, RLMClint_ul_min_mg, RHepClint, RLMClint_ml_min_kg, MLM_T_half
"""
model_names = ["GraphSAGE", "DGCNN", "GAT", "GraphNet", "ECC"]
data_names = os.listdir("/mnt/home/linjie/projects/solubility/data/classification/modelling_data_1")
for name in data_names:
    for model in model_names:
        for i in glob(f'/mnt/home/linjie/projects/solubility/result/classification/modelling_data_1/{name}/{model}/inner_5/performance.csv'):
            print(i)
        print('*'*10)
    print('*' * 100)


['CYP1A2_data_clean', 'HERG-220720_clean', 'CYP2C19_data_clean', 'CYP3A4_data_clean', 'CYP2C9_data_clean', 'CYP2D6_data_clean']
"""'CYP1A2_data_clean', 'HERG-220720_clean', CYP2C19_data_clean, CYP3A4_data_clean, CYP2C9_data_clean, CYP2D6_data_clean """
model_names = ["GraphSAGE", "DGCNN", "GAT", "GraphNet", "ECC"]
for model in model_names:
    for i in glob(f'/mnt/home/linjie/projects/solubility/result/classification/CYP_hERG_clean_data/CYP2D6_data_clean/{model}/inner_5/*'):
        print(i)
    print('*'*10)


import os
from glob import glob
import shutil
base_dir = '/mnt/home/linjie/projects/solubility/data/classification/modelling_data_1'
dataset_names = os.listdir(base_dir)
dataset_names.remove('modelling_data_1.csv')
for i in dataset_names:
    try:
        shutil.copytree(os.path.join('/mnt/home/linjie/projects/solubility/result/classification/modelling_data_1', i, 'DMPNN'), os.path.join('/home/linjie/Downloads/solubility', i, 'DMPNN'))
    except:
        pass
    # os.makedirs(os.path.join('/home/linjie/Downloads/solubility', i))

dataset_names = list(set(dataset_names)-set(['modelling_data_1.csv', 'HHep_T_half', 'MHep_T_half', 'RHep_T_half']))



base_dir = '/mnt/home/linjie/projects/solubility/result/classification/CYP_hERG_clean_data'
dataset_names = os.listdir(base_dir)
dataset_names.remove('modelling_data_1.csv')

data_paths = glob(f"{base_dir}/*_data_clean")
for i in data_paths:
    test_scores_path = glob(f'{i}')
import pickle
label_encoder_path = '/mnt/home/linjie/projects/solubility/data/classification/modelling_data_1/HLM_T_half/HLM_T_half.pkl'
with open(label_encoder_path, 'rb') as f:
    le = pickle.load(f)
    class_labels = le.classes_
    print(class_labels)


dataset_names = ['HHep_T_half', 'HLMClint_ul_min_mg', 'RLMClint_ul_min_mg', 'RHepClint', 'RLMClint_ml_min_kg', 'MLM_T_half',
 'F_data', 'MHep_T_half', 'HHepClint', 'HLMClint_ml_min_kg', 'RHep_T_half', 'HLM_T_half', 'PAMPA_permeability', 'MDCK_ER']
model_names = ['CMPNN', 'DMPNN']
for dataset_name in dataset_names:
    for model in model_names:
        test_score_path = os.path.join(base_dir, dataset_name, model, 'test_scores.csv')
        # test_score_df = pd.read_csv(test_score_path)
        with open(test_score_path, 'r')as f:
            lines = f.readlines()
        a = "Task,Mean kappa,Standard deviation kappa,Fold 0 kappa,Fold 1 kappa,Fold 2 kappa,Fold 3 kappa,Fold 4 kappa,Mean accuracy,Standard deviation accuracy,Fold 0 accuracy,Fold 1 accuracy,Fold 2 accuracy,Fold 3 accuracy,Fold 4 accuracy,Mean recall,Standard deviation recall,Fold 0 recall,Fold 1 recall,Fold 2 recall,Fold 3 recall,Fold 4 recall,Mean mcc,Standard deviation mcc,Fold 0 mcc,Fold 1 mcc,Fold 2 mcc,Fold 3 mcc,Fold 4 mcc,Mean f1,Standard deviation f1,Fold 0 f1,Fold 1 f1,Fold 2 f1,Fold 3 f1,Fold 4 f1,Mean Medium_precision,Standard deviation Medium_precision,Fold 0 Medium_precision,Fold 1 Medium_precision,Fold 2 Medium_precision,Fold 3 Medium_precision,Fold 4 Medium_precision,Mean Low_precision,Standard deviation Low_precision,Fold 0 Low_precision,Fold 1 Low_precision,Fold 2 Low_precision,Fold 3 Low_precision,Fold 4 Low_precision,Mean High_precision,Standard deviation High_precision,Fold 0 High_precision,Fold 1 High_precision,Fold 2 High_precision,Fold 3 High_precision,Fold 4 High_precision\n"
        a+=lines[1]
        with open(test_score_path, 'w') as f:
            f.write(a)


dataset_names = ['HHep_T_half', 'HLMClint_ul_min_mg', 'RLMClint_ul_min_mg', 'RHepClint', 'RLMClint_ml_min_kg', 'MLM_T_half',
 'F_data', 'MHep_T_half', 'HHepClint', 'HLMClint_ml_min_kg', 'RHep_T_half', 'HLM_T_half', 'PAMPA_permeability', 'MDCK_ER']
model_names = ['CMPNN', 'DMPNN']
for dataset_name in dataset_names:
    for model in model_names:
        test_score_path = os.path.join(base_dir, dataset_name, model, 'test_scores.csv')
        # test_score_df = pd.read_csv(test_score_path)
        with open(test_score_path, 'r')as f:
            lines = f.readlines()
        a = "Task,Mean kappa,Standard deviation kappa,Fold 0 kappa,Fold 1 kappa,Fold 2 kappa,Fold 3 kappa,Fold 4 kappa,Mean accuracy,Standard deviation accuracy,Fold 0 accuracy,Fold 1 accuracy,Fold 2 accuracy,Fold 3 accuracy,Fold 4 accuracy,Mean recall,Standard deviation recall,Fold 0 recall,Fold 1 recall,Fold 2 recall,Fold 3 recall,Fold 4 recall,Mean mcc,Standard deviation mcc,Fold 0 mcc,Fold 1 mcc,Fold 2 mcc,Fold 3 mcc,Fold 4 mcc,Mean f1,Standard deviation f1,Fold 0 f1,Fold 1 f1,Fold 2 f1,Fold 3 f1,Fold 4 f1,Mean Medium_precision,Standard deviation Medium_precision,Fold 0 Medium_precision,Fold 1 Medium_precision,Fold 2 Medium_precision,Fold 3 Medium_precision,Fold 4 Medium_precision,Mean Low_precision,Standard deviation Low_precision,Fold 0 Low_precision,Fold 1 Low_precision,Fold 2 Low_precision,Fold 3 Low_precision,Fold 4 Low_precision,Mean High_precision,Standard deviation High_precision,Fold 0 High_precision,Fold 1 High_precision,Fold 2 High_precision,Fold 3 High_precision,Fold 4 High_precision\n"
        a+=lines[1]
        with open(test_score_path, 'w') as f:
            f.write(a)




base_dir = '/mnt/home/linjie/projects/solubility/result/classification/modelling_data_1'
dataset_names = os.listdir(base_dir)
dataset_names.remove('run_all.py')
dataset_names.remove('modelling_data_1.csv')
model_names = ['CMPNN', 'DMPNN']
for dataset_name in dataset_names:
    for model in model_names:
        test_scores_path = os.path.join(base_dir, dataset_name, model, 'test_scores.csv')
        test_scores_df = pd.read_csv(test_scores_path)
        test_scores_df = test_scores_df.loc[:, list(pd.Series(test_scores_df.columns).str.contains('Mean'))]
        test_scores_df.insert(loc=0, column='model', value=model)
        test_scores_df.columns = test_scores_df.columns.str.replace('Mean ', '')
        test_scores_df.to_csv(test_scores_path.replace('test_scores.csv', 'performance.csv'), index=False, float_format='%.3f')


import os
import pandas as pd
result = {}
kappa = 0
base_dir = '/mnt/home/linjie/projects/solubility/result/classification/modelling_data_1'
for idx, fname in enumerate(['RLMClint_ml_min_kg', 'RLMClint_ul_min_mg']):
    metric_path = os.path.join(base_dir,fname, 'metric_'+fname+'.csv')
    df = pd.read_csv(metric_path)
    print(pd.DataFrame(df.loc[df["kappa"].idxmax(), ['dataset', 'model', 'kappa']]).T)



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["Japan", "china", "Japan", "Korea","china"])
print('标签个数:%s' % le.classes_)
print('标签值标准化:%s' % le.transform(["Japan", "china", "Japan", "Korea","china"]))
print('标准化标签值反转:%s' % le.inverse_transform([0, 2 ,0 ,1 ,2]))