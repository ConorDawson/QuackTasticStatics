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
    p = len(X)
    if p < 3:
        return np.mean(X)  # Fallback to mean for insufficient data

    norm_diff = np.sum((X - theta0) ** 2)
    if norm_diff == 0:
        return np.mean(X)  # Prevent division by zero

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
                    if (year, month) not in data[company]:
                        data[company][(year, month)] = 0
                    data[company][(year, month)] += float(hours)  # Sum hours for each month
    except Exception as e:
        print(f"Error fetching data: {e}")
    return data

def seasonal_adjustment(hours, month, seasonal_factors):
    """
    Adjust hours based on the given month and seasonal factors.
    """
    factor = seasonal_factors.get(month, 1.0)  # Default to 1.0 if no factor is available
    if hours is None:  # Handle missing data gracefully
        return 0  # Return 0 or any other fallback value
    return hours * factor

def predict_hours_for_next_month(data, seasonal_factors, global_sigma2, global_theta0):
    """
    Predicts hours for the next month for each company using James-Stein estimator.
    """
    predictions = {}
    current_month = date.today().month
    next_month = (current_month % 12) + 1  # Calculate next month (e.g., December -> January)

    for company, monthly_data in data.items():
        # Aggregate all hours by month
        months = sorted(monthly_data.keys())
        adjusted_hours = []

        # Apply seasonal adjustment for the current month
        for (year, month), hours in monthly_data.items():
            adjusted_hours.append(seasonal_adjustment(hours, month, seasonal_factors))
        
        # Predict next month's hours with adjusted values
        predicted_value = james_stein_estimator_with_factors(
            np.array(adjusted_hours), global_sigma2, global_theta0
        )
        predictions[company] = predicted_value * seasonal_factors.get(next_month, 1.0)
    return predictions

def plot_predictions(historical_data, predictions, company_name):
    """
    Plots historical data and predicted data with seasonal factors for visualization.
    """
    if company_name not in historical_data:
        messagebox.showerror("Error", f"No data available for {company_name}.")
        return

    historical_hours = []
    months = []

    # Collect data for historical hours and months
    for (year, month), hours in sorted(historical_data[company_name].items()):
        months.append(f"{month}-{year}")
        historical_hours.append(hours)

    # Predict hours for next month
    predicted_hours = [predictions[company_name]]  # Prediction for the next month

    # X-axis: Historical data months + prediction for the next month
    months.append(f"{(date.today().month % 12) + 1}-{date.today().year}")  # Add next month for prediction
    hours = historical_hours + predicted_hours

    plt.figure(figsize=(10, 6))
    plt.plot(months[:-1], historical_hours, label="Historical Hours", marker='o', color='blue')
    plt.plot([months[-2], months[-1]], [historical_hours[-1], predicted_hours[0]], label="Predicted Next Month", marker='x', color='red')
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

    seasonal_factors = seasonal_factor.ordered_normalized_values  # Assuming this is returned from the file

    global_theta0 = np.mean([hour for company_data in historical_data.values() for hour in company_data.values()])
    global_sigma2 = np.mean([np.var(list(company_data.values())) for company_data in historical_data.values()])

    predictions = predict_hours_for_next_month(historical_data, seasonal_factors, global_sigma2, global_theta0)

    # Create tkinter GUI
    root = tk.Tk()
    root.title("Company Hours Prediction")

    # Set window size (width x height)
    root.geometry("600x400")  # Adjust size as needed

    tk.Label(root, text="Select a Company:", font=("Helvetica", 14)).pack(pady=10)

    # Create buttons for each company
    for company_name in historical_data.keys():
        btn = tk.Button(
            root,
            text=company_name,
            command=lambda name=company_name: plot_predictions(historical_data, predictions, name),
            font=("Helvetica", 12),
            width=20,  # Optional: set button width
            height=2   # Optional: set button height
        )
        btn.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
