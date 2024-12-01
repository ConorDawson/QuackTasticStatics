import os
import dask.dataframe as dd
import calendar

# Define the folder path
folder_path = "csvs/scats-reports"

# Dictionary to map filenames to months
months_map = {
    "January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr",
    "May": "May", "June": "Jun", "July": "Jul", "August": "Aug",
    "September": "Sep", "October": "Oct", "November": "Nov", "December": "Dec"
}

# Initialize a dictionary to store month and row counts
rows_per_month = {}

# Iterate through the files in the folder
for file_name in os.listdir(folder_path):
    if file_name.startswith("SCATS") and file_name.endswith("2022.csv"):
        for month, abbreviation in months_map.items():
            if abbreviation in file_name:
                file_month = month
                break
        
        # Use Dask to load the CSV file
        file_path = os.path.join(folder_path, file_name)
        try:
            df = dd.read_csv(file_path)
            rows_per_month[file_month] = len(df)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

# Calculate the average row count
total_rows = sum(rows_per_month.values())
average_rows = total_rows / len(rows_per_month)

# Calculate the normalized values
normalized_values = {month: round(count / average_rows, 2) for month, count in rows_per_month.items()}

# Ensure the output is ordered by month
ordered_months = list(calendar.month_name)[1:]  # January to December
ordered_normalized_values = {month: normalized_values.get(month, 0) for month in ordered_months}

# Print the results in order of months
print(f"Average rows: {average_rows:.2f}")
for month, normalized_value in ordered_normalized_values.items():
    print(f"{month}: {normalized_value}")
