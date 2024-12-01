import numpy as np
import psycopg2
from datetime import date
import matplotlib.pyplot as pltch

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
    Applies the James-Stein estimator with dynamic factors.
    """
    p = len(X)
    if p < 3:
        return np.mean(X)  # Fallback to the mean for insufficient data

    norm_diff = np.sum((X - theta0)**2)
    if norm_diff == 0:
        return np.mean(X)  # Prevent division by zero

    shrinkage_factor = max(0, (p - 2) * sigma2 / (norm_diff + 1e-6))  # Add small constant to avoid instability
    theta_JS = X - shrinkage_factor * (X - theta0)
    return np.mean(theta_JS)

def fetch_hours_data():
    """
    Fetches historical hours worked data from the database.
    Returns a dictionary {company_name: [hours]}.
    """
    query = "SELECT company_name, hours FROM timesheet_hours WHERE work_date <= CURRENT_DATE"
    data = {}

    try:
        # Connect to the database
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for company, hours in rows:
                    if company not in data:
                        data[company] = []
                    data[company].append(float(hours))
    except Exception as e:
        print(f"Error fetching data: {e}")
    return data

def seasonal_adjustment(hours, current_month):
    """
    Adjusts hours based on seasonality. This is a placeholder; adjust based on real-world data.
    """
    seasonal_factor = {
        1: 0.9,  # January - Less busy
        11: 1.2, # November - More busy
        12: 1.3, # December - Peak
    }
    return hours * seasonal_factor.get(current_month, 1.0)

def predict_hours_for_next_30_days(data, global_sigma2, global_theta0):
    """
    Predicts hours for the next 30 days for each company using the James-Stein estimator.
    """
    predictions = {}
    current_month = date.today().month  # Get the current month for seasonal adjustment

    for company, hours in data.items():
        # Optionally apply seasonal adjustment (if needed)
        adjusted_hours = [seasonal_adjustment(h, current_month) for h in hours]
        
        # Use the James-Stein estimator
        predictions[company] = james_stein_estimator_with_factors(
            np.array(adjusted_hours), global_sigma2, global_theta0
        )
    return predictions

def plot_predictions(historical_data, predictions):
    """
    Plots historical data and predicted data for visualization.
    """
    companies = list(predictions.keys())
    historical_means = [np.mean(historical_data[company]) for company in companies]
    predicted_means = [predictions[company] for company in companies]

    # Bar plot for historical vs predicted hours
    x = np.arange(len(companies))  # Company indices
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, historical_means, width, label="Historical Mean")
    bars2 = ax.bar(x + width/2, predicted_means, width, label="Predicted (Next 30 Days)")

    # Add labels and formatting
    ax.set_xlabel("Companies")
    ax.set_ylabel("Hours")
    ax.set_title("Historical vs Predicted Hours (Next 30 Days)")
    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=45, ha="right")
    ax.legend()

    # Display values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text by 3 points
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Fetch historical data
    historical_data = fetch_hours_data()

    # Dynamic theta0 and sigma2
    if historical_data:
        global_theta0 = np.mean([hour for hours in historical_data.values() for hour in hours])
        global_sigma2 = np.mean([np.var(hours) for hours in historical_data.values()])
    else:
        global_theta0 = 0  # Fallback if no data
        global_sigma2 = 1  # Fallback if no data

    # Predict future hours
    predictions = predict_hours_for_next_30_days(historical_data, global_sigma2, global_theta0)

    # Print predictions
    print("Predicted hours for the next 30 days:")
    for company, predicted_hours in predictions.items():
        print(f"{company}: {predicted_hours:.2f}")

    # Plot the results
    if historical_data and predictions:
        plot_predictions(historical_data, predictions)
