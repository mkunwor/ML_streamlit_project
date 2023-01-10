import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

#random seed
seed=42

#READ original dataset
iris_df=pd.read_csv('data/Iris.csv')
iris_df.sample(frac=1,random_state=seed)

#selecting features and target data
X=iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=iris_df[['Species']]

#split data into train and test sets
#70% training anf 30% test

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)

#create an instance of the random forest classifer
clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy}")

#save the model
joblib.dump(clf,"output_models/rf_model.sav")
