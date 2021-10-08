import pandas as pd

df1 = pd.read_csv('./_w4_method1.csv')
df1["label"] = 0
df2 = pd.read_csv('./merge.csv')

# 将筛选出来的填充到新的表上
df1['label'] = df1['id'].map(df2.set_index('id')['label'])
df1 = df1['label']
df1.to_csv('result.txt',index=False,header=None)
print('Done!')