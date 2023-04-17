from tqdm import tqdm
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error


def plot_y(y, title, x_label, y_label, density):
    plt.style.use('ggplot')
    plt.hist(y, bins=20, color='steelblue', edgecolor='k', density=density)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

#load data
data = pd.read_csv("data/ELD_clean/ECFP_ELD_clean.csv")
title='ECFP_ELD_clean'
x_label='activity_value'
y_label = 'num'
plot_y(data['y'], title, x_label, y_label, False)
plot_y(data['y'], title, x_label, 'probability', True)

#data StandardScaler
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=0)
plot_y(y_train, title+'_y_train', x_label, y_label, False)
plot_y(y_test, title+'_y_test', x_label, y_label, False)
# sc_x = StandardScaler()
# sc_x.fit(x_train)
# x_train_std = sc_x.transform(x_train)
# x_test_std = sc_x.transform(x_test)
# # a = sc_train.inverse_transform(x_train_std)
# sc_y = StandardScaler()
# sc_y.fit(np.array(y_train).reshape(-1,1))
# y_train_std = sc_y.transform(np.array(y_train).reshape(-1,1))
# y_test_std = sc_y.transform(np.array(y_test).reshape(-1,1))
# y_train_std = y_train_std.reshape(-1)
# y_test_std = y_test_std.reshape(-1)


from sklearn.ensemble import RandomForestRegressor
#随机森林参数优化
#优化n_estimators
# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
# 每隔10步建立一个随机森林，获得不同n_estimators的得分
score_lt = []
for i in tqdm(range(0,200,10), total=len(range(0,200,10))):
    model = RandomForestRegressor(n_estimators=i+1, random_state=0, n_jobs=cpu_count())
    score = cross_val_score(model, x_train, y_train, cv=5).mean()
    #model.fit(x_train, y_train)
    # score = model.score(x_test, y_test)
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*10+1))
#最大得分：0.7178707602996995 子树数量为：181
# 绘制学习曲线
x = np.arange(1,201,10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()


# 在41附近缩小n_estimators的范围为170-190
score_lt = []
for i in tqdm(range(170,190), total=len(range(170,190))):
    model = RandomForestRegressor(n_estimators=i+1, random_state=0, n_jobs=cpu_count())
    score = cross_val_score(model, x_train, y_train, cv=5).mean()
    # model.fit(x_train, y_train)
    # score = model.score(x_test, y_test)
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)+170))
#最大得分：0.7180735846616476 子树数量为：182
# 绘制学习曲线
x = np.arange(170,190)
plt.subplot(111)
plt.plot(x, score_lt,'o-')
plt.show()


# # 建立n_estimators为182的随机森林
# model = RandomForestRegressor(n_estimators=182, random_state=0, n_jobs=cpu_count())
# # 用网格搜索调整max_depth
# param_grid = {'max_depth': np.arange(1, 20)}
# GS = GridSearchCV(model, param_grid, cv=5)
# GS.fit(x_train, y_train)
# best_param = GS.best_params_
# best_score = GS.best_score_
# print(best_param, best_score)


# # 用网格搜索调整max_features
# param_grid = {'max_features':np.arange(32,512)}
# model = RandomForestRegressor(n_estimators=182, random_state=0, n_jobs=cpu_count())
# GS = GridSearchCV(model, param_grid, cv=5)
# GS.fit(x_train, y_train)
# best_param = GS.best_params_
# best_score = GS.best_score_
# print(best_param, best_score)
# best_model = model.best_estimator_
# y_pred = best_model.predict(x_test)
# print('score', best_model.score(x_test, y_test))


model = RandomForestRegressor(n_estimators=182, random_state=0, n_jobs=cpu_count())
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#决定系数（r2_score）
r2_score(y_test, y_pred)#0.7156
#中值绝对误差（Median absolute error）
median_absolute_error(y_test, y_pred)#0.4451
#均方差（mean squared error)
mean_squared_error(y_test, y_pred)#1.0667
#平方根误差(root mean square error)
sqrt(mean_squared_error(y_test, y_pred))#1.0328
score = model.score(x_test, y_test)#0.7180




def try_different_method(model, model_name):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = median_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f'{model_name}, r2={r2:.3f}, mae={mae:.3f}, mse={mse:.3f}, rmse={rmse:.3f}')

####决策树####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
try_different_method(model_DecisionTreeRegressor, 'DecisionTreeRegressor')
#DecisionTreeRegressor, r2=0.465, mae=0.500, mse=2.005, rmse=1.416
####线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
try_different_method(model_LinearRegression, 'LinearRegression')
#LinearRegression, r2=0.550, mae=0.739, mse=1.688, rmse=1.299
####SVM####
from sklearn import svm
model_SVR = svm.SVR()
try_different_method(model_SVR, 'SVR')
#SVR, r2=0.722, mae=0.408, mse=1.044, rmse=1.022
####NN####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_jobs=cpu_count())
try_different_method(model_KNeighborsRegressor, 'KNeighborsRegressor')
#KNeighborsRegressor, r2=0.655, mae=0.500, mse=1.295, rmse=1.138
####随机森林####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=182, random_state=0, n_jobs=cpu_count())#这里使用20个决策树
try_different_method(model_RandomForestRegressor, 'RandomForestRegressor')
#RandomForestRegressor, r2=0.718, mae=0.447, mse=1.058, rmse=1.028
####Adaboost####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
try_different_method(model_AdaBoostRegressor, 'AdaBoostRegressor')
#AdaBoostRegressor, r2=0.214, mae=0.996, mse=2.947, rmse=1.717
####GBRT####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
try_different_method(model_GradientBoostingRegressor, 'GradientBoostingRegressor')
#GradientBoostingRegressor, r2=0.455, mae=0.812, mse=2.045, rmse=1.430
####Bagging####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor(n_jobs=cpu_count())
try_different_method(model_BaggingRegressor, 'BaggingRegressor')
#BaggingRegressor, r2=0.689, mae=0.461, mse=1.165, rmse=1.079
####ExtraTree极端随机树####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
try_different_method(model_ExtraTreeRegressor, 'ExtraTreeRegressor')
#ExtraTreeRegressor, r2=0.477, mae=0.500, mse=1.960, rmse=1.400













