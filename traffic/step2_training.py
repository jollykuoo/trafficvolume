import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score as r2
import xgboost
import lightgbm
from Method.evaluate import regression_evaluate


# 加载数据
def load_data():
    data = pd.read_csv("./traffic_finished.csv")
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


# 调参
def param():
    x, y = load_data()
    x = x.values
    y = y.values

    x = MinMaxScaler().fit_transform(x)  # 标准化数据

    # xgboost调参
    depth_ = np.arange(3, 10, 1)
    lr_ = np.arange(0.05, 0.51, 0.05)
    subsample_ = np.arange(0.3, 1.1, 0.1)
    d_data = xgboost.DMatrix(x, y)
    for i in subsample_:
        print(i)
        params = {'max_depth': 30, 'eta': 0.4, 'subsample': 0.9}
        xgb = xgboost.cv(params=params, nfold=5, dtrain=d_data)
        print(xgb)

    # GBRT调参, GBDT(n_estimators=500, max_depth=4, learning_rate=0.1, min_samples_split=4)
    n_ = np.arange(50, 501, 50)
    depth_ = np.arange(3, 10, 1)
    lr_ = np.arange(0.01, 0.21, 0.05)
    split_ = np.arange(2, 10, 1)
    record_score = []
    for i in split_:
        print(i)
        score = cross_val_score(GBRT(n_estimators=500, max_depth=4, learning_rate=0.1, min_samples_split=i),
                                x, y, cv=5)
        record_score.append(score.mean())
        print(score.mean())
    plt.plot(split_, record_score)
    plt.show()

    # DT调参


# 使用调参后参数训练
def modeling():
    x, y = load_data()
    feature_names = x.columns
    x = x.values
    y = y.values

    # x = StandardScaler().fit_transform(x)  # 标准化数据

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
    print(x_train.shape, x_test.shape)

    # 1.xgboost
    print("---------xgboost--------")
    d_train = xgboost.DMatrix(x_train, y_train, feature_names=feature_names)
    d_test = xgboost.DMatrix(x_test, y_test, feature_names=feature_names)
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': ['mape', 'mae'],  # 交叉验证的metric
              'eta': 0.4,
              'subsample': 0.9,
              'max_depth': 10,
              'verbosity': 0
              }
    time0 = time.time()
    xgb = xgboost.train(params=params, num_boost_round=10, dtrain=d_train,
                        evals=[(d_train, 'tr'), (d_test, 'valid')], verbose_eval=10)
    time1 = time.time()

    ipt = xgb.get_score(importance_type="gain").values()  # 返回dict, 转化成列表
    print(*zip(feature_names, (pd.Series(ipt)/sum(ipt)).tolist()))  # 打印特征权重

    joblib.dump(xgb, "./xgb_joblib.dat")  # 保存模型

    # 调用回归评测指标模块, 进行评测
    y_pred = xgb.predict(d_test)
    regression_evaluate(y_test=y_test, y_pred=y_pred, mape=True, plot_line=False, plot_range=range(-100, 0))

    # 2.LightGBM
    print("---------lightgbm--------")
    lgb_train = lightgbm.Dataset(x_train, y_train, feature_name=feature_names.to_list(), categorical_feature='auto')
    lgb_test = lightgbm.Dataset(x_test, y_test, reference=lgb_train, feature_name=feature_names)
    params = {'objective': 'regression', 'num_iterations': 100, 'learning_rate': 0.1,  # core params
              'max_depth': 10, 'min_samples_leaf': 20, 'min_child_weight': 1e-3, 'min_split_gain': 0,
              'bagging_fraction': 0.9,  # most important objective params
              'feature_fraction': 1, 'feature_fraction_bynode': 1,  # colsample params
              'metric': ['mape', 'mae'], 'verbosity': -1
              }
    time2 = time.time()
    lgb = lightgbm.train(params, lgb_train, valid_sets=(lgb_train, lgb_test), valid_names=['train', 'test'],
                         verbose_eval=10)
    time3 = time.time()

    ipt = lgb.feature_importance(importance_type="gain")
    print(*zip(feature_names, ipt/sum(ipt)))  # 打印特征权重

    joblib.dump(lgb, "./lgb_joblib.dat")  # 保存模型

    y_pred = lgb.predict(x_test, num_iteration=lgb.best_iteration)
    regression_evaluate(y_test=y_test, y_pred=y_pred, mape=True, plot_line=False, plot_range=range(-100, 0))

    print("\nxgboost duration:%f \nlightgbm duration:%f" % (time1-time0, time3-time2))

    # # 3.DT
    # print("---------DT--------")
    # dt = DecisionTreeRegressor(max_depth=12)
    # dt.fit(x_train, y_train)
    # y_pred = dt.predict(x_test)
    #
    # # 评估模型
    # print("train_scoring:", dt.score(x_train, y_train), "\ntest_scoring:", dt.score(x_test, y_test))
    # print("R^2:", r2(y_test, y_pred))  # R^2
    # print("MSE:", MSE(y_test, y_pred))  # MSE
    # print("MAE:", MAE(y_test, y_pred))  # MAE
    # print("MAPE:", mape(y_test, y_pred)*100, "%")  # MAPE
    #
    # # 图示法
    # plt.scatter(y_test, y_pred, s=6)
    # plt.xlabel("y_test")
    # plt.ylabel("y_pred")
    # plt.show()
    #
    # # plt.plot(y_test[-100:], label="y_true")
    # # plt.plot(y_pred[-100:], label="y_pred")
    # # plt.legend()
    # # plt.show()

    # # 4.GBRT
    # print("---------GBRT--------")
    # gbrt = GBRT(n_estimators=500, max_depth=4, learning_rate=0.1, min_samples_split=4)
    # gbrt.fit(x_train, y_train)
    # y_pred = gbrt.predict(x_test)  # 逆转换
    #
    # # 评估模型
    # print("train_scoring:", gbrt.score(x_train, y_train), "\ntest_scoring:", gbrt.score(x_test, y_test))
    # print("R^2:", r2(y_test, y_pred))  # R^2
    # print("MSE:", MSE(y_test, y_pred))  # MSE
    # print("MAE:", MAE(y_test, y_pred))  # MAE
    # print("MAPE:", mape(y_test, y_pred)*100, "%")  # MAPE
    #
    # # 图示法
    # plt.scatter(y_test, y_pred, s=6)
    # plt.xlabel("y_test")
    # plt.ylabel("y_pred")
    # plt.show()
    #
    # plt.plot(y_test[-100:], label="y_true")
    # plt.plot(y_pred[-100:], label="y_pred")
    # plt.legend()
    # plt.show()

    pass


if __name__ == "__main__":
    # param()
    modeling()
    pass
