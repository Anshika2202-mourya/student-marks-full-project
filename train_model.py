import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os

def main():
    df = pd.read_csv('data/student_marks.csv')
    X = df[['hours_studied', 'attendance_pct', 'assignments_submitted', 'previous_gpa']]
    y = df['marks']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'features': X.columns.tolist()}, 'models/model.pkl')
    print(f'Trained model saved to models/model.pkl. RMSE={rmse:.3f}, R2={r2:.3f}')

if __name__ == '__main__':
    main()
