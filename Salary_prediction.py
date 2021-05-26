
import joblib;
model = joblib.load("Salary.pkl");
print(model.predict([[3.5]]));

