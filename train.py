import os
from sklearn import datasets
from model import FeedForward
from sklearn.model_selection import train_test_split

# directory to store data
data_home = "./data/"

trainSize = 64
testSize = 1000

lr = 0.7
n_epochs = 2

# make directory for data if it does not exists
if not os.path.exists(data_home):
    os.mkdir(data_home)


# load data
digits = datasets.load_digits()

X, y = digits.images, digits.target

n_samples = len(X)

# reshape data
X = X.reshape((n_samples, -1))

# normalize images
X /= 255.0

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create network
ff = FeedForward(n_inputs = 64, n_hidden = 32, n_outputs = 10)

# train network
ff.fit(X_train, y_train, lr, n_epochs)

# test network
y_hat = ff.predict(X_test)

print(y_hat)
print(y_train)

# evaluate the accuracy

