import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

land_area=[500,850,1400,1500,2000,600,3500,1200,2500,1000] #land area in sq. feet
land_price=[1500000,2500000,4000000,5500000,7000000,10000000,9000000,3500000,7500000,5000000]

#convert lists to numpy arrays and reshape for sklearn
x=np.array(land_area).reshape(-1,1) #Reshape to 2D array
y=np.array(land_price)

#create and train the model
model=LinearRegression() #k "LinearRegression" is a readymade model and it takes input in 2D array
model.fit(x,y) # Here x is 2D (x is feature and y is label , it's why x is in 2D)

# Prediction Technique
predicted_price=model.predict([[1500]]) #predict for 3500 sq.feet land
print(f"Predicted price for 3500 sq. unit : {predicted_price[0]: ,.2f}")