import numpy as np
from Model import func,CustomModelWrapper
from JobsReading import getData
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error




X, Y = getData('Jobs_By_Industry___Beginning_2012.csv')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=40)


def CustomModel():
    params, _ = curve_fit(func, xdata=X_train,ydata=Y_train.values.ravel(),p0=np.ones((len(X_train.columns)+1)))
    customModel = CustomModelWrapper(func, params)
    Y_predicted_custom = customModel.predict(X_test)

    print("\nMean absolute percentage error from Custom Model ")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_custom))
    print('Mean squared error from Custom Model')
    print(mean_squared_error(Y_test,Y_predicted_custom))

    return Y_predicted_custom

def LinearModel():
    linearModel = LinearRegression()
    linearModel.fit(X_train, Y_train)
    Y_predicted_linear = linearModel.predict(X_test)

    print("\nMean absolute percentage error from Linear Model")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_linear))
    print('Mean squared error from Linear Model')
    print(mean_squared_error(Y_test, Y_predicted_linear))


    return Y_predicted_linear

def SVRModel():
    SVRModel = SVR()
    SVRModel.fit(X_train,Y_train)
    Y_predicted_SVR = SVRModel.predict(X_test)

    print("\nMean absolute percentage error from SVR")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_SVR))
    print('Mean squared error from SVR')
    print(mean_squared_error(Y_test, Y_predicted_SVR))

    return Y_predicted_SVR

def Plots():
    YCustom = CustomModel()
    YLinear = LinearModel()
    YSVR = SVRModel()

    Range = len(Y_test)

    sns.scatterplot(x=range(0,Range), y=Y_test, color='white',edgecolor='black')
    plt.scatter(x=range(0,Range), y=YCustom, color='purple')
    plt.scatter(x=range(0,Range), y=YLinear, color='red')
    plt.scatter(x=range(0, Range), y=YSVR, color='green')

    plt.show()



