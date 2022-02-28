import pandas as pd
import numpy as np
import torch

USED_REP = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def z_score(x):
    min = np.min(x)
    max = np.max(x)
    x = (x - min) / (max - min)
    return x

#drop out the row that have the null value
#return the data and label as tensor

def read_data(path):
    titanic_data = pd.read_csv(path)
    titanic_lable = titanic_data['Survived']
    titanic_data = titanic_data[USED_REP]
    titanic_data['Age'].fillna(30, inplace=True)
    titanic_data['Fare'].fillna(0, inplace=True)
    titanic_data['Embarked'].fillna('S', inplace=True)
    # 直接drop对应indx即可删除该行

    titanic_data.replace({
        'Sex': {'male': 0, 'female': 1},
        'Embarked': {'C': 0, 'Q': 1, 'S': 2}
    }, inplace=True)

    #normalize
    age_ary = np.array(titanic_data['Age'])
    age_ary = z_score(age_ary)
    fare_ary = np.array(titanic_data['Fare'])
    fare_ary = z_score(fare_ary)

    titanic_data_a = np.array(titanic_data)
    titanic_data_a[:, 2] = age_ary
    titanic_data_a[:, 5] = fare_ary

    titanic_lable = torch.tensor(titanic_lable).long()

    titanic_data_train = torch.tensor(titanic_data_a)
    return titanic_data_train, titanic_lable

    # print(titanic_data_train, titanic_data_train.shape)
    # print(titanic_data.head())

if __name__ == "__main__":
    read_data('./data/train.csv')