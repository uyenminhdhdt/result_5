import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

df = pd.read_csv("5HK.csv")

cdf = df[['SoTcD_1','SoTcR_1','DTB_1','SoTcD_2','SoTcR_2','DTB_2','SoTcD_3','SoTcR_3','DTB_3','SoTcD_4','SoTcR_4','DTB_4','SoTcD_5','SoTcR_5','DTB_5','KetQua']]
x = cdf.iloc[:, :15]
y = cdf.iloc[:, -1]

clf = RandomForestClassifier(n_estimators=50)
clf.fit(x, y)

pickle.dump(clf, open('model.pkl', 'wb'))