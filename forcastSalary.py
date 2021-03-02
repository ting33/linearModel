#用线性回归模型预测工资
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
g=open("/Users/zhouya/Desktop/01/many_variable.csv",encoding='utf-8')
data=pd.read_csv(g)
g.close()

data['education']=data['education'].replace(['本科','研究生'],[1,2])
data['city']=data['city'].replace(['北京','上海','广州','杭州','深圳'],
                    [1,2,3,4,5])
data['title']=data['title'].replace(['P4','P5','P6','P7'],
                    [1,2,3,4])
print(data.info())
x=np.array(data[['work_length','education','title','city']])
y=np.array(data['year_salary'])
#拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#训练模型
linear=linear_model.LinearRegression()
linear.fit(X_train,y_train)

#斜率
print("斜率：{}".format(linear.coef_) )
#截距
print("截距:{}".format(linear.intercept_))
#判定系数r^2，越大越好
#训练集r^2
print("训练集合上R^2 = {:.3f}".format(linear.score(X_train,y_train)))
#测试集r^2
print("测试集合上R^2 = {:.3f}".format(linear.score(X_test,y_test)))
#预测
print(list(linear.predict(X_test)))
#回归方程
# y=1.35+1.1*work_length+5.19*education+5.92*title+0.09*city
# 我想在3年内拿到年薪30万，那么我需要达到哪一步了？
# 工作经验3年，学历研究生，城市在北京，那么我需要达到哪个职级了？
# 30=1.35+1.1*3+5.19*2+0.09*1+5.92*x
# x=2.53
# 需要达到P6左右才能实现目标