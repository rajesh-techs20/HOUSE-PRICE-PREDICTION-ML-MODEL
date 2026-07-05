import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv("house_data.csv", engine="python")

X=df[["Area","Bedrooms","Bathrooms","Age","Parking"]]
y=df["Price"]

model=LinearRegression()
model.fit(X,y)

with open("model.pkl","wb") as file:
    pickle.dump(model,file)

print("Model is trained and saved as model.pkl")