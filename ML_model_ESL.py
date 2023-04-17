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
from sklearn.ensemble import RandomForestRegressor

def plot_y(y, title, x_label, y_label, density):
    plt.style.use('ggplot')
    plt.hist(y, bins=20, color='steelblue', edgecolor='k', density=density)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

#load data
data = pd.read_csv("data/ESL-clean-final-20220704/ECFP_ESL-clean-final-20220704.csv")
title='ESL-clean-final-20220704'
x_label='pLogS'
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


model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=cpu_count())
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
#DecisionTreeRegressor, r2=0.547, mae=0.269, mse=3.172, rmse=1.781
####线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
try_different_method(model_LinearRegression, 'LinearRegression')
#LinearRegression, r2=0.594, mae=0.924, mse=2.846, rmse=1.687
####SVM####
from sklearn import svm
model_SVR = svm.SVR()
try_different_method(model_SVR, 'SVR')
#SVR, r2=0.696, mae=0.478, mse=2.129, rmse=1.459
####NN####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_jobs=cpu_count())
try_different_method(model_KNeighborsRegressor, 'KNeighborsRegressor')
#KNeighborsRegressor, r2=0.801, mae=0.393, mse=1.390, rmse=1.179
####随机森林####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=cpu_count())#这里使用20个决策树
try_different_method(model_RandomForestRegressor, 'RandomForestRegressor')
#RandomForestRegressor, r2=0.777, mae=0.404, mse=1.564, rmse=1.251
####Adaboost####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=200)#这里使用50个决策树
try_different_method(model_AdaBoostRegressor, 'AdaBoostRegressor')
#AdaBoostRegressor, r2=-0.007, mae=2.505, mse=7.049, rmse=2.655
####GBRT####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
try_different_method(model_GradientBoostingRegressor, 'GradientBoostingRegressor')
#GradientBoostingRegressor, r2=0.600, mae=0.921, mse=2.798, rmse=1.673
####Bagging####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor(n_jobs=cpu_count())
try_different_method(model_BaggingRegressor, 'BaggingRegressor')
#BaggingRegressor, r2=0.749, mae=0.411, mse=1.759, rmse=1.326
####ExtraTree极端随机树####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
try_different_method(model_ExtraTreeRegressor, 'ExtraTreeRegressor')
#ExtraTreeRegressor, r2=0.554, mae=0.287, mse=3.123, rmse=1.767













