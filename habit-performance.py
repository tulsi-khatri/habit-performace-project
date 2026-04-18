import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


np.random.seed(42)
n = 500

sleep_hours = np.clip(np.random.normal(6.5, 1.5, n), 3, 10)
study_hours = np.clip(np.random.normal(5, 2, n), 0, 12)
screen_time = np.clip(np.random.normal(5, 2.5, n), 0, 12)
mood_score = np.random.randint(1, 6, n)
exercise = np.random.randint(0, 2, n)

performance = (
    sleep_hours * 4 +
    study_hours * 9 -
    screen_time * 2.5 +
    mood_score * 5 +
    exercise * 6 +
    (study_hours ** 1.2) +   
    np.random.normal(0, 5, n)  
)

df = pd.DataFrame({
    "sleep_hours": sleep_hours,
    "study_hours": study_hours,
    "screen_time": screen_time,
    "mood_score": mood_score,
    "exercise": exercise,
    "performance": performance
})

df.to_csv("habit_data.csv", index=False)
print("Dataset created and saved as habit_data.csv")


data = pd.read_csv("habit_data.csv")

print("\nPreview of dataset:")
print(data.head())

X = data.drop("performance", axis=1)
y = data["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_cv_score = cross_val_score(lr_model, X, y, cv=5, scoring='r2')



rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
rf_pred = best_rf.predict(X_test)

rf_cv_score = cross_val_score(best_rf, X, y, cv=5, scoring='r2')

print("\nBest Parameters for Random Forest:", grid_search.best_params_)


print("\n--- Model Comparison ---")

print("\nLinear Regression")
print("R2 Score:", round(r2_score(y_test, lr_pred), 3))
print("MSE:", round(mean_squared_error(y_test, lr_pred), 2))
print("CV Score:", round(lr_cv_score.mean(), 3))

print("\nRandom Forest (Tuned)")
print("R2 Score:", round(r2_score(y_test, rf_pred), 3))
print("MSE:", round(mean_squared_error(y_test, rf_pred), 2))
print("CV Score:", round(rf_cv_score.mean(), 3))



plt.figure()
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted Performance (Random Forest)")
plt.show()



importance = best_rf.feature_importances_

plt.figure()
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()



joblib.dump(best_rf, "habit_model.pkl")
print("\nModel saved as habit_model.pkl")


sample_input = np.array([[7, 6, 3, 4, 1]])
prediction = best_rf.predict(sample_input)[0]

print("\nPredicted Performance Score:", round(prediction, 2))

if prediction > 75:
    print("High performance expected based on current habits.")
elif prediction > 50:
    print("Moderate performance. Some improvement is possible.")
else:
    print("Low performance trend. Lifestyle improvements recommended.")