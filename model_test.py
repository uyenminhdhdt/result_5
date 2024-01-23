import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

df = pd.read_csv("5HK.csv")

cdf = df[['SoTcD_1','SoTcR_1','DTB_1','SoTcD_2','SoTcR_2','DTB_2','SoTcD_3','SoTcR_3','DTB_3','SoTcD_4','SoTcR_4','DTB_4','SoTcD_5','SoTcR_5','DTB_5','KetQua']]
x = cdf.iloc[:, :15]
y = cdf.iloc[:, -1]

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=100) #state 180 Accracy 91.3%
# Huấn luyện
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))

# Đánh giá độ chính xác
y_pred = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

#ma trận trực giao
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

#mức độ quan trọng của các thuộc tính
importances = clf.feature_importances_
for feature, importance in zip(x.columns, importances):
    print(f"{feature}: {importance}")