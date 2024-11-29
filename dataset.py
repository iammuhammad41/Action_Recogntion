import os
import csv
import re
import glob
import pandas as pd

root = '/media/song/新加卷/Action_Recognition/data'

label = []
value = []
action = []

for file in glob.glob(root + '/**/*.txt', recursive=True):
    # print(file)
    f = open(file, "r")

    if file.split('/')[-1] == 'object_pose.txt':
        for i in f.read().split('\n'):
            if i != '':
                value.append(i.split())
                #Action class from folder name
                # action_name = file.split('/')[-3]
                # action.append(action_name)
    elif file.split('/')[-1] == 'grasp_class.txt':
        for i in f.read().split('\n'):
            if i.split(':') != ['']:
                label.append(i.split(':'))
    elif file.split('/')[-1] == 'action_class.txt':
        for i in f.read().split('\n'):
            if i.split(':') != ['']:
                action.append(i.split(':'))


df_action = pd.DataFrame(action)
# print('Y:', '\n', df_action)

df_op =pd.DataFrame(value)
# print('dataframe: ','\n', df_op)

df_gc =pd.DataFrame(label)
# print('DF1','\n',  df_gc)


# print('nan values', df_action.isna().sum())
df  = df_op
df[-1]=df_gc[1]
df[-2]= df_action[1]
# print('new dataframe','\n',  df)

df.rename(columns = {0:'frame_idx', -1:'Label', -2:'ActionClass'}, inplace = True)
# print('nan values in ActionClass', df['ActionClass'].isna().sum(), 'nan values in Label', df['Label'].isna().sum())
# print('again new dataframe','\n',  df)
#
# df.dropna()

# print('again new dataframe','\n',  df)
X=df.drop(['frame_idx', 'ActionClass'],axis=1)
y=df[['ActionClass']]
# print('input ', '\n',  X)
# print('y labels', y)


# y.ActionClass.value_counts()
# print('Count Labelled Classes', y.ActionClass.value_counts())
# len(y.ActionClass.value_counts())
# print('Count Labelled Classes', len(y.ActionClass.value_counts()))


y.ActionClass = y.ActionClass.str.strip()


# In[20]:

y.ActionClass
# print('y.ActionClass', y.ActionClass)

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)


# from sklearn.preprocessing import LabelEncoder
# lb = LabelEncoder()
# y = lb.fit_transform(y)

# set(y)
# print('Labelled Classes: ', set(y))
#
# dic={}
# for i in range(len(set(y))):
#     dic[list(set(y))[i]]=list(set(lb.inverse_transform(y)))[i]
# print(len(dic))
# print('Labelled Classes Dic: ', dic)


from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
transformer = MinMaxScaler()
transformer.fit(X)
X=transformer.transform(X)


Y = pd.to_numeric(y.ActionClass, downcast='signed')


Y=Y.to_numpy()

len(set(Y))
# print('Total no of classes', set(Y))
# print('Total no of classes', len(set(Y)))

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
Y = lb.fit_transform(Y)


dic={}
for i in range(len(set(Y))):
    dic[list(set(Y))[i]]=list(set(lb.inverse_transform(Y)))[i]
# print(len(dic))
# print('Labelled Classes Dic: ', dic)


X=X.astype(float)


# print(X.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)





