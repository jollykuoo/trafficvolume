import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # , mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

'''
These methods mainly use for evaluating the capacity of models, according to my commonly use, separate for two parts.
One is for classification, indicators of this kind of model contain acc, precision, recall, f1score, roc and auc.
Another is for regression, indicators of regression model consist of r2, mse, mae, mape or some graphic methods.
'''


def classify_evaluate(model=None,
                      x_test=None,
                      y_test=None,
                      y_pred=None,
                      accuracy=True,
                      precision=True,
                      recall=True,
                      f1=False,
                      conf_matrix=False,
                      roc=False,
                      auc_=False,
                      ):
    if model is not None and x_test is not None:
        y_pred = model.predict(x_test)
    elif y_pred is not None:
        y_pred = y_pred
    else:
        print("[Warning] model or y_pred is None")
        raise ValueError

    if accuracy is True:
        print("accuracy:", round(accuracy_score(y_test, y_pred), 6))

    if conf_matrix is True:
        print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))

    if len(set(y_test)) < 3:
        if precision is True:
            print("precision:", round(precision_score(y_test, y_pred), 6))

        if recall is True:
            print("recall:", round(recall_score(y_test, y_pred), 6))

        if f1 is True:
            print("f1_score:", round(f1_score(y_test, y_pred), 6))

        if auc_ or roc:
            fpr, tpr, thres = roc_curve(y_test, y_pred)
            if auc_ is True:
                print("auc:", auc(fpr, tpr))
            if roc is True:
                print("fpr:", fpr)
                print("tpr:", tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr)
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    return (np.abs(y_pred - y_true) / y_true).mean()


def regression_evaluate(model=None,
                        x_test=None,
                        y_test=None,
                        y_pred=None,
                        r2=True,
                        mse=True,
                        mae=True,
                        mape=False,
                        plot_line=False,
                        scatter=False,
                        plot_range=None,  # 作图范围, 使用[-n, 0]方式
                        # plot_intervals=False,  # 作上下浮动范围
                        # plot_anomalies=False,  # 作异常值
                        ):
    if model is not None and x_test is not None:
        y_pred = model.predict(x_test)
    elif y_pred is not None:
        y_pred = y_pred
    else:
        print("[Warning] model or y_pred is None")
        raise ValueError

    if r2 is True:
        r2_val = round(r2_score(y_test, y_pred), 6)  # R2
        print("R2:", r2_val)

    if mse is True:
        mse_val = round(mean_squared_error(y_test, y_pred), 6)  # MSE
        print("MSE:", mse_val)

    if mae is True:
        mae_val = round(mean_absolute_error(y_test, y_pred), 6)  # MAE
        print("MAE:", mae_val)

    if mape is True:
        mape_val = round(mean_absolute_percentage_error(y_test, y_pred)*100, 3)  # MAPE
        print("MAPE:", str(mape_val)+"%")

    # present by graph
    if plot_line or scatter:
        plt.figure(figsize=(10, 6))
        if plot_line:
            if plot_range is None:  # 是否需要规定范围作线
                # ********** 需完善interval上下限的计算标准 **********
                # if plot_intervals:
                #     deviation = y_pred.std()
                #     mae_val = mean_squared_error(y_test, y_pred)
                #     scale = 1
                #     lower = y_pred - (mae_val + scale * deviation)
                #     upper = y_pred + (mae_val + scale * deviation)
                #     plt.plot(lower, "r--", label="Upper Bond / Lower Bond", alpha=0.5)
                #     plt.plot(upper, "r--", alpha=0.5)
                #
                #     if plot_anomalies:
                #         anomalies = np.array([np.nan] * len(y_test))
                #         anomalies[y_test < lower] = y_test[y_test < lower]
                #         anomalies[y_test > upper] = y_test[y_test > upper]
                #         plt.plot(anomalies, "o", markersize=10, label="Anomalies")
                plt.plot(y_test, '-', label="Sample")
                plt.plot(y_pred, '--', label="Predict")
            else:
                # y_test_range = y_test[plot_range]
                # y_pred_range = y_pred[plot_range]
                # if plot_intervals:
                #     deviation = y_pred_range.std()
                #     mae_val = mean_absolute_error(y_test_range, y_pred_range)
                #     scale = 1
                #     lower = y_pred_range - (mae_val + scale * deviation)
                #     upper = y_pred_range + (mae_val + scale * deviation)
                #     plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
                #     plt.plot(upper, "r--", alpha=0.5)
                #
                #     if plot_anomalies:
                #         anomalies = np.array([np.nan] * len(y_test_range))
                #         anomalies[y_test_range < lower] = y_test_range[y_test_range < lower]
                #         anomalies[y_test_range > upper] = y_test_range[y_test_range > upper]
                #         plt.plot(anomalies, "o", markersize=10, label="Anomalies")
                plt.plot(y_test[plot_range], '-', label="Sample")
                plt.plot(y_pred[plot_range], '--', label="Predict")
            plt.legend()
            plt.show()

        if scatter:
            plt.scatter(y_test, y_pred, s=6)
            plt.xlabel("Sample")
            plt.ylabel("Predict")
            plt.show()
