import csv
import numpy as np
import pandas as pd

df1 = pd.read_table('./train2_robert_result.txt',header=None)
df2 = pd.read_table('./train2_nezha_result.txt',header=None)
df3 = pd.read_table('./train2_skep_result.txt',header=None)
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
df1.to_csv('./train2_robert_result.csv', index=False)
df2.to_csv('./train2_nezha_result.csv', index=False)
df3.to_csv('./train2_skep_result.csv', index=False)

# 筛选多份提交结果文件中预测相同的部分
result_path   = 'train2_robert_result.csv'
result_file = open(result_path,'r',encoding='utf-8')
result_reader = csv.reader(result_file)
f1 = {}
for rows in result_reader:
    f1[rows[0]] = rows[1].encode('utf-8').decode('utf-8')

forest_path = 'train2_nezha_result.csv'
forest_file = open(forest_path, 'r',encoding='utf-8')
forest_reader = csv.reader(forest_file)
f2 = {}
for rows in forest_reader:
    f2[rows[0]] = rows[1].encode('utf-8').decode('utf-8')

forest_path1 = 'train2_skep_result.csv'
forest_file1 = open(forest_path1, 'r',encoding='utf-8')
forest_reader1 = csv.reader(forest_file1)
f3 = {}
for rows in forest_reader1:
    f3[rows[0]] = rows[1].encode('utf-8').decode('utf-8')

# 进行筛选，取多模型预测相同的部分
x = set(f1.items()).intersection(set(f2.items())).intersection(set(f3.items()))
x = pd.DataFrame(x)
x.columns = ["id", "label"]
x[~df1['label'].isin(['label'])]

# 拼接text_a,label为标签数据
t1 = x
t2 = pd.read_table('./test.txt',header=None)
t2.columns = ["text_a"]
# 添加id索引
list = []
for i in range(1,83600):
    list.append(i)
id = pd.DataFrame(list)
t2['id'] = id
t2 = t2[['id','text_a']]

t1['id'] = t1['id'].astype(str)
t1['label'] = t1['label'].astype(str)
t2['id'] = t2['id'].astype(str)
t2['text_a'] = t2['text_a'].astype(str)

t3 = pd.merge(t1, t2[['id', 'text_a']], on='id', how='left')
t3 = t3[['text_a','label']]
t3.to_csv('newtest1.csv',index=False,sep=',')
print('Done!')