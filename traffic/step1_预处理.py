import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))


def to_datetime():
    data = pd.read_csv("~/Downloads/Metro_Interstate_Traffic_Volume.csv")
    print(data.info())
    print(data.describe(include="all"))
    data['date'] = data['date_time'].apply(lambda x: x.split(' ')[0])
    data['date_time'] = pd.to_datetime(data['date_time'])

    data['year'] = data['date_time'].apply(lambda x: x.year)
    data['month'] = data['date_time'].apply(lambda x: x.month)
    data['dayofweek'] = data['date_time'].apply(lambda x: x.dayofweek+1)
    data['dayofmonth'] = data['date_time'].apply(lambda x: x.day)
    data['isweekend'] = data['date_time'].apply(lambda x: int(x.dayofweek > 4))
    data['hour'] = data['date_time'].apply(lambda x: x.hour)

    data.to_csv("./traffic.csv", index=False)


def datetime_volume():
    data = pd.read_csv("./traffic.csv")
    print(data.describe().to_string())

    # 1.按hour统计
    # # 1.1统计数据量
    # sns.countplot(x="hour", data=data)  # 数据量均匀
    # plt.show()

    # # 1.2统计交通流量
    # sns.lineplot(x="hour", y="traffic_volume", data=data)  # 7-17点处于一个高峰阶段
    # plt.show()

    # # 2.按dayofweek统计
    # # # 2.1统计数据量
    # # sns.countplot(x="dayofweek", data=data)  # 数据量均匀
    # # plt.show()
    #
    # # 2.2统计交通量
    # sns.lineplot(x="dayofweek", y="traffic_volume", data=data)  # 周一到周五交通量较大, 周末交通量很低
    # plt.show()

    # # 3.按month统计
    # # # 3.1数据量统计
    # # sns.countplot(x="month", data=data)  # 数据量分布在3500-4500附近
    # # plt.show()
    #
    # # 3.2统计交通量
    # sns.lineplot(x="month", y="traffic_volume", data=data)  # 4-10月较高, 全年范围在3000-3500之间
    # plt.show()

    # # 4.按year统计
    # # # 4.1数据量统计
    # # sns.countplot(x="year", data=data)  # 2013、2016、2017、2018年数据量比较充足
    # # plt.show()
    #
    # # 4.2统计数据量
    # sns.lineplot(x="year", y="traffic_volume", data=data)  # 比较均衡
    # plt.show()

    # 5. dayofmonth
    sns.lineplot(x="dayofmonth", y="traffic_volume", data=data)
    plt.show()

    pass


# 离散值处理
def distraction():
    data = pd.read_csv("./traffic.csv")  # 修改holiday数据前数据为traffic.csv
    print(data.describe(include="O").to_string())

    # # 1.holiday
    # # 1.1数据量统计
    # sub = data.loc[data['holiday'] != 'None', :]
    # sns.countplot(y='holiday', data=sub)  # 节假日数据字段较少, 原因是节假日只在当天第一条数据说明
    # plt.show()

    # 1.2补充节假日数据
    holiday_date = data.loc[data['holiday'] != 'None', 'date']
    holiday = data.loc[data['holiday'] != 'None', 'holiday']
    d = dict(zip(holiday_date.values, holiday.values))
    data['holiday'] = data['date'].apply(lambda x: d[x] if x in d.keys() else 'None')  # 修正日期对应的节日
    data.to_csv("./traffic_modifiedholiday.csv", index=False)

    # # 1.3统计假期交通量
    # sub = data.loc[data['holiday'] != 'None', :]
    # sns.barplot(x="holiday", y="traffic_volume", data=sub)
    # plt.xticks(rotation="45")
    # plt.show()

    # # 2.weather_main
    # # # 2.1统计天气量
    # # sns.countplot(y="weather_main", data=data)  # clouds和clear居多
    # # plt.show()
    #
    # # 2.2统计不同天气下交通量
    # sns.barplot(x="weather_main", y="traffic_volume", data=data)
    # plt.show()

    # 3.其他处理
    data['isholiday'] = data['holiday'].apply(lambda x: 1 if x != 'None' else 0)
    data['weekend or holiday'] = data['isweekend'] ^ data['isholiday']  # 周末或假期
    data.to_csv("./traffic_modifiedholiday.csv", index=False)

    pass


# 使用modifiedholiday后的数据继续进行异常值处理
def continual():
    data = pd.read_csv("./traffic_modifiedholiday.csv")
    data['temp'] = data['temp'].apply(lambda x: x-273.15)  # 开尔文转换为摄氏度
    dsc = data.describe()
    print(data.info())
    print(dsc.to_string())

    # 1.temp
    Q1 = dsc.loc['25%', 'temp']
    Q3 = dsc.loc['75%', 'temp']
    IQR = Q3 - Q1
    # print(data.loc[(data['temp'] < Q1-1.5*IQR) | (data['temp'] > Q3+1.5*IQR), :])  # 查看异常值

    # # 处理前箱线图
    # sns.boxplot(data=data, x="temp")
    # plt.show()

    # 处理异常值
    data.drop(index=data.loc[(data['temp'] < Q1-1.5*IQR) | (data['temp'] > Q3+1.5*IQR), :].index, inplace=True)

    # # 处理后箱线图
    # sns.boxplot(data=data, x="temp")
    # plt.show()

    # 2.rain_1h
    # # 处理前箱线图
    # sns.boxplot(x="rain_1h", data=data)
    # plt.show()

    # Q1 = dsc.loc['25%', 'rain_1h']
    # Q3 = dsc.loc['75%', 'rain_1h']
    # IQR = Q3 - Q1
    # print(data.loc[(data['rain_1h'] < Q1 - 1.5 * IQR) | (data['rain_1h'] > Q3 + 1.5 * IQR), :])  # 异常值, 数据量过多

    # # 使用密度估计图查看, 由于异常值过多不可全部删除
    # sns.distplot(x=data['rain_1h'], rug=True)  # 从结果可知, rain_1h基本集中分布在60以内
    # plt.show()

    # print(data.loc[data['rain_1h'] > 60, :].to_string())
    data.drop(index=data.loc[data['rain_1h'] > 60, :].index, inplace=True)  # 删除异常值

    # # 处理后箱线图
    # sns.boxplot(x="rain_1h", data=data)
    # plt.show()

    # # 3.snow_1h
    # # 处理前箱线图
    # sns.boxplot(x="snow_1h", data=data)  # 由于比较均匀, 符合日常生活经验, 不需要作其他处理
    # plt.show()

    # # 4.clouds_all
    # # 处理前箱线图
    # sns.boxplot(x="clouds_all", data=data)  # 无异常值, 不需要处理
    # plt.show()

    # # 5.traffic_volume
    # sns.violinplot(x="traffic_volume", data=data)
    # plt.show()

    # # 6.热力图, 对连续型数据进行相关性分析
    # cont = data[data.describe().columns]
    # sns.heatmap(data=cont.corr())  # 连续特征之间没有显著相关性
    # plt.show()

    data.to_csv("./traffic_delete_abnormal.csv", index=False)

    pass


# 分割一天内时间块
def split_hour(hour):
    if hour in [0, 1, 2, 3, 4, 5]:
        return "Night"
    elif hour in [6, 7, 8, 9, 10]:
        return "Morning"
    elif hour in [11, 12, 13, 14]:
        return "Noon"
    elif hour in [15, 16, 17, 18]:
        return "Afternoon"
    else:
        return "Evening"


# 进行特征集构建
def feature():
    data = pd.read_csv("./traffic_delete_abnormal.csv")

    # 2.一天内时间hour分块
    data['day'] = data['hour'].apply(lambda x: split_hour(x))

    # 2.weather热编码
    weather = pd.get_dummies(data['weather_main'])

    # 3.周几day进行热编码
    hour = pd.get_dummies(data['day'])

    # 4.对week热编码
    week = pd.get_dummies(data['dayofweek'])

    # 5.组合特征
    data = pd.concat([data, weather, hour, week], axis=1)
    data = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'month', 'dayofmonth', 'isweekend', 'isholiday', 'hour',
                 'Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm',
                'Afternoon', 'Evening', 'Morning', 'Night', 'Noon',
                 'traffic_volume']]  # 选取最终变量&调整列顺序
    data.to_csv("./traffic_finished.csv", index=False)  # 保存预处理文件


if __name__ == "__main__":
    # to_datetime()
    # datetime_volume()  # 关于时间对车流量影响分析
    # distraction()  # 处理离散值
    # continual()  # 处理连续的异常值
    feature()

    pass
