import numpy as np
from sklearn.model_selection import train_test_split



def hyp(theta0,theta1, X):
    return (theta0*X + theta1)

def summation_function(thetas,X,Y,Xj=[]):
    print(Xj)
    summation = 0
    for i in range(Y.size):
        if(len(Xj)>0):
            summation += ((Y[i]-hyp(thetas[0],thetas[1],X[i]))*Xj[i])
        else:
            summation += ((Y[i]-hyp(thetas[0],thetas[1],X[i])))

    return summation


def derivative_cost_function_theta1(theta,X,Y):
    summation = summation_function(theta,X,Y,X)
    cost = (1/X.size)*summation
       
    return cost

def derivative_cost_function_theta0(theta,X,Y):
    cost = (1/X.size)*summation_function(theta,X,Y)
    return cost






def batch_gradient(x, y, Theta, learning_rate = 0.001):
    for i in range(len(Theta)):
        if(i==0):
            Theta[0] = Theta[0] - learning_rate*derivative_cost_function_theta0(Theta, x, y)
        else:
            Theta[1] = Theta[1] - learning_rate*derivative_cost_function_theta1(Theta, x, y)

    return Theta






def main():
    X = np.arange(0,5,0.1, dtype=np.float32)
    delta = np.random.uniform(-1,1, size=X.shape[0])
    Y = .4 * X + 3 + delta

    X_train, Y_train, X_test, Y_test = train_test_split(X,Y, test_size=1/3)


    Theta = [0,0]

    newTheta = batch_gradient( X_train, Y_train, Theta)

    Y_pred = hyp(newTheta[0],newTheta[1],X_test)
    print(Y_pred)
    
    Y_error = np.zeros(len(Y_pred))
    for i in range(len(Y_test)):
        Y_error[i] = Y_test[i] - Y_pred[i]

    print(Y_error)
    




if __name__ == "__main__":
    main()