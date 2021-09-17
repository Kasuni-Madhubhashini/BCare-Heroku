import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')

df['diagnosis'] = df['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
df['diagnosis'] = df['diagnosis'].astype("float64")

print(df.head(7))
df = np.array(df)
X = df[1:, 1:-1]
y = df[1:, -1]
y = y.astype('float')
X = X.astype('float')

print(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# test model's accuracy on the training data set
print('Model Training Accuracy = ', log_reg.score(X_train, y_train))

# test model's accuracy on the test data set
cm = confusion_matrix(y_test, log_reg.predict(X_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

# print the confusion matrix
print(cm)


# print the model's accuracy on test data
print('Model Test Accuracy = {}'.format((TP+TN) / (TP+TN+FN+FP)))

pickle.dump(log_reg, open('Model.pkl', 'wb'))
model = pickle.load(open('Model.pkl', 'rb'))
