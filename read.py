import pandas as pd 

df1 = pd.read_table('./train4_robert_result.txt',header=None)
df2 = pd.read_table('./train4_nezha_result.txt',header=None)
df3 = pd.read_table('./train4_skep_result.txt',header=None)

df1.columns = ['label']
df2.columns = ['label']
df3.columns = ['label']

# 添加id索引
list = []
for i in range(1,83600):
    list.append(i)

id = pd.DataFrame(list)

df1['id'] = id
df1 = df1[['id','label']]

df2['id'] = id
df2 = df2[['id','label']]

df3['id'] = id
df3 = df3[['id','label']]

# 保存结果文件
df1.to_csv('./_w4_method1.csv', index=False)
df2.to_csv('./_w3_method2.csv', index=False)
df3.to_csv('./_w2_method3.csv', index=False)