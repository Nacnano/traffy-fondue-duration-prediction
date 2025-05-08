import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from pythainlp.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATH = '/kaggle/input/traffy-fondue-dsde-chula-dataset/bangkok_traffy.csv'
SEED = 42
np.random.seed(SEED)
print("Using device: CPU (for traditional models)")

# --- Data Loading and Preprocessing ---
print("\n--- Loading Data ---")
try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except FileNotFoundError:
    print(f"CRITICAL ERROR: Data file not found at {DATA_PATH}")
    exit()

# Filter resolved tickets
df = df[df['state'] == 'เสร็จสิ้น'].copy()
print(f"Shape after filtering resolved tickets: {df.shape}")

# Convert timestamps and filter for 2025
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_localize(None)
df['last_activity'] = pd.to_datetime(df['last_activity'], errors='coerce').dt.tz_localize(None)
df = df[df['timestamp'].dt.year == 2025]
df = df.dropna(subset=['timestamp', 'last_activity'])
df['duration_days'] = (df['last_activity'] - df['timestamp']).dt.total_seconds() / 86400
df = df[df['duration_days'] > 0]
print(f"Shape after filtering for 2025 and calculating positive durations: {df.shape}")

# Cap outliers at 10th-90th percentile
lower_bound = df['duration_days'].quantile(0.10)
upper_bound = df['duration_days'].quantile(0.90)
df = df[(lower_bound <= df['duration_days']) & (df['duration_days'] <= upper_bound)]
print(f"Shape after capping durations at (~{lower_bound:.2f} days) and (~{upper_bound:.2f} days): {df.shape}")

# Text feature engineering
text_cols = ['type', 'organization', 'comment', 'address', 'subdistrict', 'district', 'province']
for col in text_cols:
    df[col] = df[col].fillna('').astype(str)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['combined_text'] = (
    "ประเภท: " + df['type'].apply(clean_text) +
    " | หน่วยงาน: " + df['organization'].apply(clean_text) +
    " | ปัญหา: " + df['comment'].apply(clean_text) +
    " | แขวง: " + df['subdistrict'].apply(clean_text) +
    " | เขต: " + df['district'].apply(clean_text)
)

# Additional features
df['issue_specificity'] = df['type'].apply(lambda x: 1 if x != '{}' else 0)
df['comment_length'] = df['comment'].str.len()
priority_types = ['ความปลอดภัย', 'จราจร']
df['is_priority'] = df['type'].apply(lambda x: 1 if any(t in x for t in priority_types) else 0)
top_orgs = df['organization'].value_counts().head(10).index
df['organization_encoded'] = df['organization'].apply(lambda x: x if x in top_orgs else 'Other')
df = pd.get_dummies(df, columns=['organization_encoded'], prefix='org')

# Temporal features
df['creation_month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Drop rows with empty comments
df = df[df['comment'].str.strip() != '']
print(f"Shape after dropping empty comments: {df.shape}")

# Split into train, validation, and test sets
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
val_size_relative = 0.1 / 0.9  # To make val 10% of original df
train_df, val_df = train_test_split(train_val_df, test_size=val_size_relative, random_state=SEED)
print(f"Train set shape: {train_df.shape} ({len(train_df)/len(df)*100:.2f}%)")
print(f"Validation set shape: {val_df.shape} ({len(val_df)/len(df)*100:.2f}%)")
print(f"Test set shape: {test_df.shape} ({len(test_df)/len(df)*100:.2f}%)")

# Scale numerical features
duration_scaler = MinMaxScaler()
train_df['duration_scaled'] = duration_scaler.fit_transform(train_df[['duration_days']])
val_df['duration_scaled'] = duration_scaler.transform(val_df[['duration_days']])
test_df['duration_scaled'] = duration_scaler.transform(test_df[['duration_days']])

length_scaler = MinMaxScaler()
train_df['comment_length_scaled'] = length_scaler.fit_transform(train_df[['comment_length']])
val_df['comment_length_scaled'] = length_scaler.transform(val_df[['comment_length']])
test_df['comment_length_scaled'] = length_scaler.transform(test_df[['comment_length']])

# --- Traditional Models Preparation ---
print("\n--- Preparing Data for Traditional Models ---")

def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm')

vectorizer = TfidfVectorizer(tokenizer=thai_tokenizer, max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_df['combined_text'])
X_val_tfidf = vectorizer.transform(val_df['combined_text'])
X_test_tfidf = vectorizer.transform(test_df['combined_text'])

other_features = ['issue_specificity', 'comment_length_scaled', 'is_priority', 'creation_month', 'day_of_week'] + \
                 [col for col in train_df.columns if col.startswith('org_')]
X_train_other = train_df[other_features].values
X_val_other = val_df[other_features].values
X_test_other = test_df[other_features].values

# Convert to dense arrays for compatibility with all models
X_train = np.hstack([X_train_tfidf.toarray(), X_train_other])
X_val = np.hstack([X_val_tfidf.toarray(), X_val_other])
X_test = np.hstack([X_test_tfidf.toarray(), X_test_other])

y_train = train_df['duration_scaled'].values
y_val = val_df['duration_scaled'].values
y_test = test_df['duration_scaled'].values

# --- Train and Evaluate Traditional Models ---
print("\n--- Training and Evaluating Traditional Models ---")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=SEED)
}

for name, model in models.items():
    print(f"\nTraining {name}")
    model.fit(X_train, y_train)
    
    # Validation set evaluation
    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = duration_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    y_val_true = duration_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    val_mae = mean_absolute_error(y_val_true, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    print(f"{name} - Val MAE: {val_mae:.2f} days, Val RMSE: {val_rmse:.2f} days")
    
    # Test set evaluation
    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = duration_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_true = duration_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    print(f"{name} - Test MAE: {test_mae:.2f} days, Test RMSE: {test_rmse:.2f} days")

# --- Visualization for XGBoost on Test Set ---
y_test_pred = duration_scaler.inverse_transform(models['XGBoost'].predict(X_test).reshape(-1, 1)).flatten()
y_test_true = duration_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test_true, y=y_test_pred, alpha=0.6)
plt.plot([min(y_test_true), max(y_test_true)], [min(y_test_true), max(y_test_true)], 'r--')
plt.title('Actual vs. Predicted Duration (Days) on Test Set - XGBoost')
plt.xlabel('Actual Duration (Days)')
plt.ylabel('Predicted Duration (Days)')
plt.grid(True)
plt.savefig('scatter_plot_test_xgboost.png')