import pandas as pd
import os


def getData(importfile):
    filename ='Jobs_By_Industry___Beginning_2012_rows.csv'
    if not os.path.isfile(filename):
        print('Getting data from file')
        df = pd.read_csv(importfile,sep=',')
        for column in df.columns:
            df = df[df[column].notnull()]
        df.sample(n = 1000)
        df.to_csv(filename,index = False)
        print('Downloaded')
    else:
        print(f'File {filename} already exists')

    X = pd.read_csv(filename, sep=',')
    Y = X['Jobs']
    X = X.drop(['Jobs'],axis = 1)

    X = pd.get_dummies(X)

    return X, Y



