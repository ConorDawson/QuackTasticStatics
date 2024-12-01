import pandas as pd
import random
from datetime import date, timedelta

# Initialize variables
start_date = date(2024, 1, 1)
end_date = date(2024, 11, 28)  # Today's date
clients = ["Tech Innovators", "Global Solutions", "Creative Minds", "Green Energy Inc.", 
           "Health Plus", "Future Vision", "Sky High Enterprises", "Urban Architects", "Digital World"]

# Generate a list of all weekdays between start_date and end_date
weekdays = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1) if (start_date + timedelta(days=x)).weekday() < 5]

# Create data
data = []
record_id = 1

for day in weekdays:
    for client in clients:
        hours = random.randint(0, 10)  # Random hours between 0 and 10
        data.append((record_id, client, hours, day))
        record_id += 1

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Record ID", "Client", "Hours", "Date"])

# Save to CSV for easy handling
df.to_csv("./csvs/work_hours.csv", index=False)
print(df)
