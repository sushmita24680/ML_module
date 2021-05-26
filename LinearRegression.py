import pandas;
dataset = pandas.read_csv('Salary_Data.csv');
x = dataset['YearsExperience'].values.reshape(30,1);
y = dataset['Salary'];
dataset.info();
from sklearn.linear_model import LinearRegression;
model = LinearRegression();
print(model.fit(x , y));
print(model.coef_);
print(model.intercept_);
print(model.predict([[2.5]]));
import joblib;
print(joblib.dump(model ,"Salary.pkl"));


