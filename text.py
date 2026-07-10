import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("house_data.csv",engine="python")

X=df[["Area","Bedrooms","Bathrooms","Age","Parking"]]
y=df["Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                                               
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Mean Square Error:",mse)
print("R2 Score:",r2)


with open("model.pkl","wb") as file:
    pickle.dump(model,file)

print("Model is trained,evaluated successfully and saved as model.pkl")
