# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd

df = pd.read_csv('parkinsons.csv')
df=df.dropna()
df.head()


selected_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
output_feature = 'status'

X=df[selected_features]
y=df[output_feature]


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=8)
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred=model.predict(X_test)
acurracy=accuracy_score(y_test,y_pred)
print(acurracy)

import joblib

joblib.dump(model, 'my_model.joblib')
