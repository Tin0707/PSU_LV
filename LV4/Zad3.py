import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622*np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0, 1, len(y))
    return y_noisy

x = np.linspace(1, 10, 50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

degrees = [2, 6, 15]
MSEtrain = []
MSEtest = []
plt.figure()

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(x)

    np.random.seed(12)
    indeksi = np.random.permutation(len(x_poly))
    i_train = indeksi[:int(0.7*len(x_poly))]
    i_test = indeksi[int(0.7*len(x_poly)):]

    xtrain = x_poly[i_train]
    ytrain = y_measured[i_train]
    xtest = x_poly[i_test]
    ytest = y_measured[i_test]

    model = LinearRegression()
    model.fit(xtrain, ytrain)

    ytrain_pred = model.predict(xtrain)
    ytest_pred = model.predict(xtest)

    MSEtrain.append(mean_squared_error(ytrain, ytrain_pred))
    MSEtest.append(mean_squared_error(ytest, ytest_pred))

    y_model = model.predict(x_poly)
    plt.plot(x, y_model, label=f'degree {deg}')

plt.plot(x, y_true, 'k--', label='true function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("MSEtrain:", MSEtrain)
print("MSEtest:", MSEtest)
