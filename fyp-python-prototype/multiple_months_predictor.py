import os
import numpy as np
import psycopg2
from datetime import date
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import seasonal_factor  # Import your seasonal factors script

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
        # Fallback with slight upward trend when data is insufficient
        return np.mean(X) + 0.01 * np.std(X)

    norm_diff = np.sum((X - theta0) ** 2)
    shrinkage_factor = max(0, (p - 2) * sigma2 / (norm_diff + 1e-6))  # Stability adjustment
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

def predict_hours_for_next_month(data, seasonal_factors, global_sigma2, global_theta0, num_months):
    """
    Predicts hours for the next `num_months` months for each company using the James-Stein estimator.
    """
    predictions = {}
    current_month = date.today().month
    current_year = date.today().year

    for company, monthly_data in data.items():
        # Sort the historical data by (year, month)
        historical_data = sorted(monthly_data.items())
        adjusted_hours = [seasonal_adjustment(hours, month, seasonal_factors)
                          for (year, month), hours in historical_data]

        company_predictions = []
        next_data = adjusted_hours.copy()

        # Generate predictions for the specified number of months
        for i in range(num_months):
            predicted_value = james_stein_estimator_with_factors(
                np.array(next_data), global_sigma2, global_theta0
            )

            # Determine the next month and year (ensure predictions start from next month)
            next_month = (current_month + i) % 12 + 1
            next_year = current_year + (current_month + i) // 12

            # Apply seasonal adjustment to the prediction
            seasonal_factor = seasonal_factors.get(next_month, 1.0)
            predicted_value *= seasonal_factor

            # Add slight randomness to avoid flat predictions
            predicted_value += np.random.normal(0, global_sigma2**0.5 * 0.1)

            company_predictions.append((next_year, next_month, predicted_value))
            next_data.append(predicted_value)

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

    global_theta0 = np.mean([hour for company_data in historical_data.values() for hour in company_data.values()])
    global_sigma2 = np.mean([np.var(list(company_data.values())) for company_data in historical_data.values()])

    # Create tkinter GUI
    root = tk.Tk()
    root.title("Company Hours Prediction")

    tk.Label(root, text="Enter number of months to predict:", font=("Helvetica", 12)).pack(pady=5)
    num_months_entry = tk.Entry(root, font=("Helvetica", 12), width=10)
    num_months_entry.pack(pady=5)

    def predict_and_plot(company_name):
        try:
            num_months = int(num_months_entry.get())
            if num_months < 1:
                raise ValueError("Number of months must be at least 1.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of months.")
            return

        predictions = predict_hours_for_next_month(historical_data, seasonal_factors, global_sigma2, global_theta0, num_months)
        plot_predictions(historical_data, predictions, company_name, num_months)

    for company_name in historical_data.keys():
        tk.Button(root, text=company_name,
                  command=lambda name=company_name: predict_and_plot(name),
                  font=("Helvetica", 12), width=20, height=2).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
