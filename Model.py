

def func(X, *args):
    res = args[len(X.columns)]
    for i in range(len(X.columns)):
        res += X[X.columns[i]] * args[i]
    return res

class CustomModelWrapper:
    def __init__(self,predicateFun,params):
        self.predicateFun = predicateFun
        self.params = params

    def predict(self,X):
        return self.predicateFun(X,*self.params)