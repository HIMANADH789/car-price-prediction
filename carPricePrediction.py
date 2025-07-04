import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("carPrice.csv")
df = df.dropna()
df = df.drop(columns=["CarName", "car_ID"])

X = df.drop("price", axis=1)
y = np.log1p(df["price"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

base_learners = [
    ("rf", RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42)),
    ("gb", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]
meta_learner = Ridge()

stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner, cv=5)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", stacking_model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

mae = mean_absolute_error(y_test_exp, y_pred_exp)
rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_exp, y=y_pred_exp, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

joblib.dump(model_pipeline, "car_price_model.pkl")
