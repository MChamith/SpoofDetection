import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

print('collecting train data')

with open("train_data.pickle", "rb") as myfile:
    X_train = pickle.load(myfile)
    X_label = pickle.load(myfile)
print('train data collected')

with open('test_data.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
    test_label = pickle.load(handle)
    X_val = pickle.load(handle)
    val_label = pickle.load(handle)
scaler = StandardScaler()
# Fit on training set only.
print('Scaling')
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
print('Scale transformed')

model = LinearSVC(C=2.0, max_iter=50000, verbose=1)
print('starting to fit model')
model.fit(X_train, X_label)
print('model fitted')
score = model.score(X_val, val_label)
print('score = ' + str(score))