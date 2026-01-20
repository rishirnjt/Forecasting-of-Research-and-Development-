
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox


df = pd.read_csv("/Users/rishiranjit/Downloads/expenditure.csv")

numeric_cols = [
    'Business enterprise', 'Government', 'Higher Education',
    'Private non-profit', 'Rest of the world', 'Basic research',
    'Applied research', 'Experimental development', 'Medical and health sciences'
]

df[numeric_cols] = df[numeric_cols].fillna(0)
df = df.dropna(subset=['TIME', 'Country'])
df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
df = df.dropna(subset=['TIME'])
df['Total_RnD'] = df[numeric_cols].sum(axis=1)
df['Log_RnD'] = np.log1p(df['Total_RnD'])
df = df.sort_values(['Country', 'TIME']).reset_index(drop=True)

# Create lag features
df['Lag_1'] = df.groupby('Country')['Total_RnD'].shift(1).fillna(0)
df['Lag_2'] = df.groupby('Country')['Total_RnD'].shift(2).fillna(0)

features = ['TIME', 'Lag_1', 'Lag_2']
target = 'Log_RnD'

# Train models per country and store them
models = {}

for country in df['Country'].unique():
    country_data = df[df['Country'] == country].copy()
    X = country_data[features]
    y = country_data[target]

    if len(X) < 3:
        continue  # skip countries with too few data points

    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
    xgb = XGBRegressor(n_estimators=300, random_state=42).fit(X, y)

    # Evaluate models 
    y_pred_rf = rf.predict(X)
    rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))

    # Choose best model (lowest RMSE)
    models[country] = rf  # you can also compare lr, rf, xgb

def predict_rnd():
    country = country_var.get()
    if country not in models:
        messagebox.showerror("Error", f"No model available for {country}")
        return

    model = models[country]
    country_data = df[df['Country'] == country].copy().sort_values('TIME')
    last_vals = country_data['Total_RnD'].values[-2:]  # last 2 years

    future_years = [2026, 2027, 2028, 2029, 2030]
    future_preds = []

    lag_1, lag_2 = last_vals[-1], last_vals[-2]

    for year in future_years:
        X_future = pd.DataFrame({
            'TIME': [year],
            'Lag_1': [lag_1],
            'Lag_2': [lag_2]
        })
        log_pred = model.predict(X_future)[0]
        pred = np.expm1(log_pred)
        future_preds.append(pred)
        lag_2 = lag_1
        lag_1 = pred

    result_text = f"Future R&D Forecast for {country} (2026-2030):\n"
    for y_val, p_val in zip(future_years, future_preds):
        result_text += f"{y_val}: {p_val:,.2f}\n"

    result_label.config(text=result_text)

# GUI window
# ================= GUI WINDOW =================
root = tk.Tk()
root.title("R&D Forecasting")

# Set window size
window_width = 600
window_height = 400

# Center window on screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Main centered frame
main_frame = tk.Frame(root)
main_frame.pack(expand=True)

# Heading
heading = tk.Label(
    main_frame,
    text="Forecast R&D Expenditure of Country",
    font=("Arial", 16, "bold")
)
heading.pack(pady=20)

# Country selection
tk.Label(main_frame, text="Select Country:", font=("Arial", 11)).pack(pady=5)

country_var = tk.StringVar()
country_dropdown = ttk.Combobox(
    main_frame,
    textvariable=country_var,
    values=list(models.keys()),
    width=35,
    state="readonly"
)
country_dropdown.pack(pady=5)
country_dropdown.current(0)

# Forecast button
predict_button = tk.Button(
    main_frame,
    text="Forecast R&D",
    font=("Arial", 11),
    command=predict_rnd
)
predict_button.pack(pady=15)

# Result display
result_label = tk.Label(
    main_frame,
    text="",
    justify="left",
    font=("Arial", 11)
)
result_label.pack(pady=10)

root.mainloop()
