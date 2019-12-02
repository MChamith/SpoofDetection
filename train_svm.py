import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

with open("train_data.pickle", "rb") as myfile:
    X_train = pickle.load(myfile)
    X_label = pickle.load(myfile)

with open('test_data.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
    test_label = pickle.load(handle)
    X_val = pickle.load(handle)
    val_label = pickle.load(handle)
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

model = LinearSVC(C=100.0, random_state=42)
model.fit(X_train, X_label)
score = model.score(X_val, val_label)
print('score = ' + str(score))