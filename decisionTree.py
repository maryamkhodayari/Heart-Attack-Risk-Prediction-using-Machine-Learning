#ایمپورت کتابخانه های مورد نیاز
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

#خواندن دیتاست و تبدیل به دیتافریم
ha = pd.read_csv("heartattack.csv")


#data  cleaning
ha = ha.drop_duplicates()

#تقسیم متغیر های مستقل و وابسته
X = ha.drop(columns="output")
y = ha["output"]

#افراز دیتاست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

#آموزش مدل
model = DecisionTreeClassifier(criterion="entropy", max_depth=3 ,min_samples_split = 2 ,min_samples_leaf = 1, random_state = 3)
model.fit(X_train, y_train)

#تست مدل
risk = model.predict(X_test)

#بررسی جزئیات عملکرد مدل
print("Predictions:", risk[:5])
print("test:", y_test[:5])
print("Accuracy:", model.score(X_test, y_test))
print(classification_report(y_test,risk))

#رسم confusion matrix
cm = confusion_matrix(y_test, risk)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

#دریافت ویژگی های کاربر جدید
print("Enter patient information forda prediction:")

age = int(input("age: "))
sex = int(input("sex (0 = female, 1 = male): "))
cp = int(input("cp / chest pain type (normaly 0 to 3): "))
trtbps = int(input("trtbps / resting blood pressure (normaly 94 to 200): "))
chol = int(input("chol / cholesterol (normaly 126 to 564): "))
fbs = int(input("fbs / fasting blood sugar (0 or 1): "))
restecg = int(input("restecg (normaly 0 to 2): "))
thalachh = int(input("thalachh / max heart rate (normaly 71 to 202): "))
exng = int(input("exng / exercise induced angina (0 or 1): "))
oldpeak = float(input("oldpeak (normaly 0.0 to 6.2): "))
slp = int(input("slp / slope (normaly 0 to 2): "))
caa = int(input("caa / number of major vessels (normaly 0 to 4): "))
thall = int(input("thall (normaly 0 to 3): "))

#ذخیره اطلاعات کاربر جدید
new_patient = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]],
columns=X.columns)

#پیش بینی مدل درباره کابر
prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Result: High chance of heart attack")
else:
    print("Result: Low chance of heart attack")