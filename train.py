import os
import numpy as np
from sklearn.metrics import  accuracy_score
import matplotlib.pyplot as plt
from sklearn import datasets
from model import FeedForward
from sklearn.model_selection import train_test_split

# directory to store data
data_home = "./data/"
imgs_home = "./imgs/"

lr = 0.5
n_epochs = 50

# make directory for data if it does not exists
if not os.path.exists(data_home):
    os.mkdir(data_home)

if not os.path.exists(imgs_home):
    os.mkdir(imgs_home)

# load data
digits = datasets.load_digits()

X, y = digits.images, digits.target

n_samples = len(X)

# reshape data
X = X.reshape((n_samples, -1))
y = y.reshape((-1, 1))

# normalize images
X /= 255.0

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

n_hidden = 4

# create network
ff = FeedForward(n_inputs = 64, n_hidden = n_hidden, n_outputs = 10)

# train network
ff.fit(X_train, y_train, lr, n_epochs)

# test network
y_hat = ff.predict(X_test)

score = accuracy_score(y_test, y_hat) * 100.0

print(f"Accuracy: {score:.2f}")

plt.plot(np.arange(len(ff.train_err)), ff.train_err, color="blue", label="Training error")
plt.title("Training error vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig(imgs_home + "ff_" + f"{n_hidden}_hidden.png")