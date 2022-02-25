import torch
import torch.nn as nn
from config import *
import pandas as pd
import numpy as np
USED_REP = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
from main import TitanicPredictor, read_data, make_batches, load_checkpoint

def z_score(x):
    min = np.min(x)
    max = np.max(x)
    x = (x - min) / (max - min)
    return x

def read_data(path):
    titanic_data = pd.read_csv(path)
    # titanic_lable = titanic_data['Survived']
    titanic_data = titanic_data[USED_REP]
    titanic_data['Age'].fillna(30, inplace=True)
    titanic_data['Fare'].fillna(0, inplace=True)
    titanic_data['Embarked'].fillna('S', inplace=True)

    # 直接drop对应indx即可删除该行

    titanic_data.replace({
        'Sex': {'male': 0, 'female': 1},
        'Embarked': {'C': 0, 'Q': 1, 'S': 2}
    }, inplace=True)
    # titanic_data.drop(titanic_data[np.isnan(titanic_data['Embarked'])].index, inplace=True)

    #normalize
    age_ary = np.array(titanic_data['Age'])
    age_ary = z_score(age_ary)
    fare_ary = np.array(titanic_data['Fare'])
    fare_ary = z_score(fare_ary)

    titanic_data_a = np.array(titanic_data)
    titanic_data_a[:, 2] = age_ary
    titanic_data_a[:, 5] = fare_ary

    # titanic_lable = torch.tensor(titanic_lable).long()

    titanic_data_train = torch.tensor(titanic_data_a)
    return titanic_data_train

    # print(titanic_data_train, titanic_data_train.shape)
    # print(titanic_data.head())

def divide_p():
    data = read_data('./data/test.csv')
    # test_num=test_num #approximately 10%
    #
    # data_temp=torch.split(data,[data.shape[0]-test_num,test_num],dim=0)
    # train_data=data_temp[0]
    test_data=data

    # lable_temp=torch.split(lable,[lable.shape[0]-test_num,test_num],dim=0)
    # train_lable=lable_temp[0].float()
    # test_lable=lable_temp[1].float()

    dataset={'test':{'data':test_data, 'batches':make_batches(test_data)}}
    return dataset

def predict():
    TP = TitanicPredictor(args).to(args.device)
    TP = load_checkpoint(TP,
        './checkpoint/cp-epoch-50-time-2022-02-25 01:11:56.429743.pth.tar')
    dataset = divide_p()
    test_data = dataset['test']['data']
    batches = dataset['test']['batches']

    predict = []
    for start, end in batches:
        input = test_data[start:end]
        with torch.no_grad():
            output = TP(input)

        for prob in list(output):
            if prob >= 0.5:
                predict.append(1)
            else:
                predict.append(0)

    return predict

# li = [[1, 'cheng', 'A','True'], [2, 'wei', 'B','False'], [3, 'wang', 'C','False']]
# columns = ["index", "usename", "team",'admin']
# dt = pd.DataFrame(li, columns=columns)
# dt.to_excel("save_usename_team.xlsx", index=0)
# # dt.to_csv("result_csv.csv", index=0)



if __name__ == '__main__':
    pre = predict()
    result = []
    for i, p in enumerate(pre):
        result.append([892+i, 1])
    columns = ['PassengerId', 'Survived']
    dt = pd.DataFrame(result, columns=columns)
    dt.to_csv("result_csv.csv", index=0)