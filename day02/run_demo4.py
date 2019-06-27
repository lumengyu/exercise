import numpy as np
import pandas as pd

df = pd.DataFrame(data = np.arange(12).reshape(3,4),index=list('abc'),columns=list('xyzw'),dtype=np.float16)
print(df,type(df))
print(df.index,df.columns)
print(df.values,type(df.values))

#import seaborn as sns
#df = sns.load_dataset("tips")
#print(df,type(df))
#df.to_csv(path_or_buf="../data/abc.csv",index=True,index_label="row_id")

df = pd.read_csv("../data/tips.csv")
df.info()
df.head()
print("小费金额与小费总金额是否存在相关性")
#相关性应该是散点图，df默认操作是列
total_bill = df["total_bill"]
tip = df["tip"]
import matplotlib.pyplot as plt
#指定x与y的值
plt.scatter(total_bill,tip)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('total_bill')
plt.ylabel('tips')
plt.title('消费与小费的散点图')
plt.show()

print("性别与小费总金额是否存在相关性")
#对筛选后的df里面的tip列（series类型），求平均
#真的保留，假的去除
female_mean = df[df['sex'] == 'Female']['tip'].mean()
male_mean = df[df['sex'] == 'Male']['tip'].mean()
#用柱状图显示
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.bar(['female','male'],[female_mean,male_mean])
plt.title('性别与小费的散点图')
plt.show()

print("每周就餐日期的消费金额")
#index就是分组的列
ss = df.groupby(by='day')['total_bill'].sum()
print(ss,type(ss))
plt.bar(ss.index,ss.values,color='#ffbbdd')
plt.show()

print("就餐类型所占的百分比")
#value_counts 这个函数只能用于series，该函数的功能是对当前列进行分组然后求记录数（count）
#返回也是series
ss = df['time'].value_counts()
#print(ss,type(ss))
plt.pie(x=ss.values,labels=ss.index,autopct='%.2f%%')
plt.show()




