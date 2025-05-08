import pandas as pd
import numpy as np
from datetime import datetime

# Load data
DATA_PATH = '/kaggle/input/traffy-fondue-dsde-chula-dataset/bangkok_traffy.csv'
df = pd.read_csv(DATA_PATH)

# Convert timestamps to datetime, removing timezone information
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_localize(None)
df['last_activity'] = pd.to_datetime(df['last_activity'], errors='coerce').dt.tz_localize(None)

# Filter resolved tickets (assuming 'state' == 'เสร็จสิ้น' for completed tickets)
resolved_state = 'เสร็จสิ้น'  # Adjust based on your dataset
df_resolved = df[df['state'] == resolved_state].copy()

# Check if there are resolved tickets
print("Number of resolved tickets:", len(df_resolved))
if df_resolved.empty:
    raise ValueError(f"No tickets found with state '{resolved_state}'. Check unique states: {df['state'].unique()}")

# Calculate duration in days
df_resolved['duration_days'] = (df_resolved['last_activity'] - df_resolved['timestamp']).dt.total_seconds() / 86400

# Drop rows with invalid durations
df_resolved = df_resolved.dropna(subset=['duration_days'])
df_resolved = df_resolved[df_resolved['duration_days'] >= 0]

# Extract year from timestamp
df_resolved['year'] = df_resolved['timestamp'].dt.year

# Compute overall summary statistics
print("\nOverall Summary Statistics for Resolution Durations (Days):")
print(df_resolved['duration_days'].describe().to_string())

# Compute yearly summary statistics
yearly_summary = df_resolved.groupby('year').agg({
    'duration_days': [
        'count', 'mean', 'median', 'std', 'min', 'max',
        lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)
    ]
}).rename(columns={'<lambda_0>': '25%', '<lambda_1>': '75%'})

# Compute proportion of long cases (>365 days)
long_cases_yearly = df_resolved[df_resolved['duration_days'] > 365].groupby('year').size()
total_cases_yearly = df_resolved.groupby('year').size()
proportion_long = (long_cases_yearly / total_cases_yearly).fillna(0)

# Compute number of long cases and extreme cases (>730 days)
extreme_cases_yearly = df_resolved[df_resolved['duration_days'] > 730].groupby('year').size()

# Add to yearly summary
yearly_summary['proportion_long'] = proportion_long
yearly_summary['num_long_cases'] = long_cases_yearly
yearly_summary['num_extreme_cases'] = extreme_cases_yearly.fillna(0)

# Print yearly summary
print("\nYearly Summary Statistics for Resolution Durations (Days):")
print(yearly_summary.to_string())

# Save yearly summary to CSV for further analysis
yearly_summary.to_csv('/kaggle/working/yearly_duration_summary.csv')