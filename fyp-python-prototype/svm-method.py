import os
import numpy as np
import psycopg2
from datetime import date
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestRegressor  # New model for better prediction
import seasonal_factor  # Assuming seasonal_factor is available

# Database connection information
DB_CONFIG = {
    "user": "postgres",
    "password": "test",
    "host": "localhost",
    "port": 5432,
    "database": "postgres"
}

def james_stein_estimator_with_factors(X, sigma2, theta0):
    """
    Applies the James-Stein estimator for better predictions.
    """
    p = len(X)
    if p < 3 or np.sum((X - theta0) ** 2) == 0:
        return np.mean(X) + 0.01 * np.std(X)

    norm_diff = np.sum((X - theta0) ** 2)
    shrinkage_factor = max(0, (p - 2) * sigma2 / (norm_diff + 1e-6))  
    theta_JS = X - shrinkage_factor * (X - theta0)
    return np.mean(theta_JS)

def fetch_hours_data():
    """
    Fetches historical hours worked data grouped by month and year from the database.
    """
    query = """
    SELECT company_name, EXTRACT(MONTH FROM work_date) AS month, EXTRACT(YEAR FROM work_date) AS year, SUM(hours) 
    FROM timesheet_hours 
    WHERE work_date <= CURRENT_DATE
    GROUP BY company_name, month, year
    ORDER BY company_name, year, month
    """
    data = {}

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for company, month, year, hours in rows:
                    if company not in data:
                        data[company] = {}
                    data[company][(int(year), int(month))] = float(hours)
    except Exception as e:
        print(f"Error fetching data: {e}")
    return data

def seasonal_adjustment(hours, month, seasonal_factors):
    """
    Adjusts hours based on the given month and seasonal factors.
    """
    factor = seasonal_factors.get(month, 1.0)  # Default to 1.0 if no factor is available
    return hours * factor

def predict_hours_for_next_month(data, seasonal_factors, num_months):
    """
    Predicts hours for the next `num_months` months for each company using Random Forests.
    """
    predictions = {}
    current_month = date.today().month
    current_year = date.today().year

    for company, monthly_data in data.items():
        # Sort the historical data by (year, month)
        historical_data = sorted(monthly_data.items())
        
        # Prepare features and target (hours worked)
        features = []
        target = []
        for (year, month), hours in historical_data:
            # Add additional feature: month number and historical hours of last few months
            features.append([(year - current_year) * 12 + (month - current_month), month] + 
                            [hours])  # Lagged feature: the hours worked
            target.append(hours)

        features = np.array(features)
        target = np.array(target)

        # Train the Random Forest model (using more trees for better performance)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features, target)

        company_predictions = []
        for i in range(num_months):
            next_month = (current_month + i) % 12 + 1
            next_year = current_year + (current_month + i) // 12
            feature = np.array([[ (next_year - current_year) * 12 + (next_month - current_month), next_month] + 
                                [np.mean(target)]])  # Lagged feature (use average of past hours for simplicity)

            # Predict the value for the next month
            predicted_value = rf.predict(feature)[0]

            # Apply seasonal adjustment to the prediction
            seasonal_factor = seasonal_factors.get(next_month, 1.0)
            predicted_value *= seasonal_factor

            # Add slight randomness to avoid flat predictions
            predicted_value += np.random.normal(0, 0.1)

            company_predictions.append((next_year, next_month, predicted_value))

        predictions[company] = company_predictions
    return predictions

def plot_predictions(historical_data, predictions, company_name, num_months):
    """
    Plots historical data and predicted data with seasonal factors for visualization.
    """
    if company_name not in historical_data:
        messagebox.showerror("Error", f"No data available for {company_name}.")
        return

    historical_hours = []
    months = []

    # Collect historical hours and months
    for (year, month), hours in sorted(historical_data[company_name].items()):
        months.append(f"{month}-{year}")
        historical_hours.append(hours)

    # Add predictions to the data, ensuring predictions start after the current month
    for year, month, predicted_hours in predictions[company_name]:
        months.append(f"{month}-{year}")
        historical_hours.append(predicted_hours)

    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(months[:len(months) - num_months], historical_hours[:len(months) - num_months],
             label="Historical Hours", marker='o', color='blue')
    plt.plot(months[len(months) - num_months:], historical_hours[len(months) - num_months:],
             label="Predicted Hours", linestyle='--', marker='x', color='red')
    plt.xlabel("Month")
    plt.ylabel("Hours")
    plt.title(f"Hours Trend for {company_name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def main():
    historical_data = fetch_hours_data()

    if not historical_data:
        messagebox.showerror("Error", "No historical data found.")
        return

    seasonal_factors = seasonal_factor.ordered_normalized_values  # Assuming this is returned from your seasonal factors file

    # Create tkinter GUI
    root = tk.Tk()
    root.title("Company Hours Prediction")

    tk.Label(root, text="Enter number of months to predict:", font=("Helvetica", 12)).pack(pady=10)
    num_months_entry = tk.Entry(root, font=("Helvetica", 12))
    num_months_entry.pack(pady=5)

    def on_predict_button_click():
        company_name = company_combobox.get()
        try:
            num_months = int(num_months_entry.get())
            if num_months <= 0:
                raise ValueError
            predictions = predict_hours_for_next_month(historical_data, seasonal_factors, num_months)
            plot_predictions(historical_data, predictions, company_name, num_months)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number of months.")

    tk.Label(root, text="Select Company:", font=("Helvetica", 12)).pack(pady=10)
    company_combobox = ttk.Combobox(root, values=list(historical_data.keys()), font=("Helvetica", 12))
    company_combobox.pack(pady=5)
    
    predict_button = tk.Button(root, text="Predict", font=("Helvetica", 12), command=on_predict_button_click)
    predict_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
