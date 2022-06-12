import pandas as pd
import os


def getData(importfile):
    filename ='Jobs_By_Industry___Beginning_2012_rows.csv'
    if not os.path.isfile(filename):
        print('Getting data from file')
        df = pd.read_csv(importfile, sep=',')
        for column in df.columns:
            df = df[df[column].notnull()]
        dictionary = df.to_dict('records')
        helper = []
        for row in dictionary:
            if row['Year'] == 2012:
                helper.append(row['Jobs'])
            else:
                for secondrow in dictionary:
                    if int(row['Year'])-1 == int(secondrow['Year']) and row['Region'] == secondrow['Region'] and row['Industry'] == secondrow['Industry']:
                        helper.append(secondrow['Jobs'])
        df['PrevJobs'] = helper
        df = df.sample(n=2000)
        df.to_csv(filename,index=False)
        print('Downloaded')
    else:
        print(f'File {filename} already exists')

    X = pd.read_csv(filename, sep=',')
    Y = X['Jobs']
    X = X.drop(['Jobs'],axis = 1)

    X = pd.get_dummies(X)

    print(Y)
    print(X)

    return X, Y



