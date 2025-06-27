# House-Price-Prediction
📊 Problem Statement
Goal: Predict the price per square foot of a property based on meaningful numerical features after filtering out noise, outliers, and irrelevant categorical data.

🧹 Data Cleaning & Preprocessing
Steps followed:

Removed Irrelevant Features:
Dropped columns: area_type, society, balcony, availability (no correlation with price).

Cleaned size column:
Converted values like "2 BHK" to numeric (e.g., 2).
Cleaned total_sqft column:
Converted ranges like 2100-2850 to average value.
Dropped non-numeric and malformed entries.

Created price_per_sqft:
Calculated: price_per_sqft = price * 100000 / total_sqft
Reduced high cardinality in location:
Grouped locations with fewer than 10 listings into 'other'

Removed outliers:
Used IQR to eliminate outliers in price_per_sqft.
Applied custom logic to filter unrealistic total_sqft per BHK (minimum 140 sqft/bedroom + 200 sqft margin).

Handled missing values:
Filled missing bath values with mode.
Dropped remaining nulls.

📈 Feature Selection
After cleaning:
Kept only numerical and significant features:
size, total_sqft, bath, price_per_sqft

🤖 Models Used
Linear Regression
KNN Regressor
Random Forest Regressor
Trained using a train-test split (90% test) due to consistent high performance and minimal variance.

✅ Final Model: Random Forest Regressor
n_estimators=10, random_state=42
Train R² Score: ~1.00
Test R² Score: ~0.80

Slight overfitting but acceptable due to well-structured data.

📉 Evaluation Metrics
MSE: 97843.7777493347
R2 Score: 0.973370990323183
(MSE and R² values printed via sklearn.metrics.)
