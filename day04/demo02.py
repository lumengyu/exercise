#One-Hot编码：可以把文字映射到数字的编码方式
#有几种类型就有几位，同一时间只有一位为1

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
# 说所有的样本加起来必须保证所有列的特征值都要遍历到
enc.fit([['男', '中国', '足球'],
['女', '美国', '篮球'],
['男', '日本', '羽毛球'],
['女', '中国', '乒乓球']]) # 这里一共有4个数据，3种特征
array = enc.transform([['男', '美国', '乒乓球']]).toarray() # 这里使用一个新的数据来测
print(array) # [[ 1 0 0 1 0 0 0 0 1]]
print(enc.inverse_transform(array))