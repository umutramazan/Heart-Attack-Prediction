#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

#read data
data=pd.read_csv("heart.csv")

#data info
print(data.info())
print(data.describe())

# Explore correlations between features
correlation_matrix=data.corr()

plt.figure(figsize=(14,10))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.show()


data.corr()["output"].sort_values().plot(kind="bar")

#data visualization

numeric_list = ["age", "trtbps", "chol", "thalachh", "oldpeak", "output"]
df_numeric = data.loc[:, numeric_list]
sns.pairplot(df_numeric, hue = "output", diag_kind = "kde")
plt.show()

#model building

x=data.iloc[:,0:13]
y=data.iloc[:,-1:]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#scale features
sc=StandardScaler() 

X_train=sc.fit_transform(x_train) 
X_test=sc.transform(x_test)

#implementation of the model

logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

#predict

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

#evalation
cm=confusion_matrix(y_test,y_pred)
print(cm)

print("Test accuracy: {}".format(accuracy_score(y_pred,y_test)))