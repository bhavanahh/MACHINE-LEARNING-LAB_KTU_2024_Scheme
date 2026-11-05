
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_csv('auto-mpg.csv')
data.replace("?",np.nan,inplace=True)
data.dropna(inplace=True)
data["displacement"]=data["displacement"].astype(float)

x=data[["displacement"]]
y=data["mpg"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

linear_model=LinearRegression()
linear_model.fit(x_train,y_train)
y_pred_linear=linear_model.predict(x_test)

linear_mse=mean_squared_error(y_test,y_pred_linear)
linear_r2_score=r2_score(y_test,y_pred_linear)

poly2=PolynomialFeatures(degree=2)
x_train_poly2=poly2.fit_transform(x_train)
x_test_poly2=poly2.transform(x_test)

poly_model2=LinearRegression()
poly_model2.fit(x_train_poly2,y_train)
y_pred_poly2=poly_model2.predict(x_test_poly2)

poly2_mse=mean_squared_error(y_test,y_pred_poly2)
poly2_r2_score=r2_score(y_test,y_pred_poly2)

poly3=PolynomialFeatures(degree=3)
x_train_poly3=poly3.fit_transform(x_train)
x_test_poly3=poly3.transform(x_test)

poly_model3=LinearRegression()
poly_model3.fit(x_train_poly3,y_train)
y_pred_poly3=poly_model3.predict(x_test_poly3)

poly3_mse=mean_squared_error(y_test,y_pred_poly3)
poly3_r2_score=r2_score(y_test,y_pred_poly3)

comparison=pd.DataFrame({
    "Model":["Linear Regression","Polynomial Regression (Degree 2)","Polynomial Regression (Degree 3)"],
    "MSE":[round(linear_mse,2),round(poly2_mse,2),round(poly3_mse,2)],
    "R2 Score":[round(linear_r2_score,2),round(poly2_r2_score,2),round(poly3_r2_score,2)]
})

print("\n comparison")
print(comparison)
x_plot=pd.DataFrame(np.linspace(x["displacement"].min(),x["displacement"].max(),300),columns=["displacement"])

x_plot_poly2=poly2.transform(x_plot)
y_plot=poly_model2.predict(x_plot_poly2)

plt.figure(figsize=(8,6))
plt.scatter(x["displacement"],y,color="blue",label="Actual Data")
plt.plot(x_plot["displacement"],y_plot,color="red",label="polynomial regression")

plt.xlabel("displacement")
plt.ylabel("mpg")
plt.title("polynomial Regression")
plt.legend()
plt.savefig("poly")
plt.show()
