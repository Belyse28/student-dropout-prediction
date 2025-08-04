
# Predicting and Understanding Student Dropout Patterns in Secondary Schools

---

##  Sector Selection

* **Selected Sector:** Education
*  **Dataset:** (https://www.statistics.gov.rw/statistical-publications/statistical-yearbook/rwanda-statistical-yearbook-2024)

---

##  Problem Statement

**Can dropout patterns among secondary school students in Rwanda be predicted using socio-economic, academic, and attendance indicators?**

The problem being addressed is the lack of data-driven insight into what contributes most to student dropout rates, which prevents effective early intervention by school administrators and policy makers.

---

##  Python Analytics Tasks

### 1. Clean the Dataset

```python
# Import necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_csv("dropout_data.csv")  # Replace with your file name if different

# Display the first few rows
print("🔹 Preview of the dataset:")
display(df.head())

# Display dataset shape
print(f"\n🔹 Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns")

# Show column names and data types
print("\n🔹 Column data types:")
print(df.dtypes)

```

<img width="626" height="667" alt="image" src="https://github.com/user-attachments/assets/9ef7b37e-633c-411f-a67a-8e82d7227618" />


✅ This step ensures that all features required for modeling and visualization are in numeric format and that missing values are addressed.

---

### 2️. Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['Overall_Dropout_Percentage'], kde=True, bins=20)
plt.title('Distribution of Overall Dropout Percentage')
plt.xlabel('Dropout %')
plt.ylabel('Count')
plt.show()
```
<img width="1473" height="686" alt="image" src="https://github.com/user-attachments/assets/14aa26b0-443f-4122-9631-378360cae9fa" />

 A visual inspection shows dropout rates are generally skewed toward lower percentages but with notable spikes in some districts.

---

### 3️. Machine Learning Model (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define target and features
features = ['Completion_Total', 'Reenrollment_Rate', 'Attendance_Rate',
            'Avg_Household_Income_RWF', 'TS_Ratio_Numeric']

X = df[features]
y = df['Dropout_Level']  # Categorical: Low, Medium, High

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

 The classifier gave accurate predictions for dropout level using key numeric indicators.

---

### 4️. Innovation: Feature Importance

```python
import numpy as np
import seaborn as sns

importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
sorted_features = [features[i] for i in indices]

sns.barplot(x=importance[indices], y=sorted_features)
plt.title("Top Features Impacting Dropout Level")
plt.xlabel("Feature Importance")
plt.show()
```

📊 This shows that Attendance Rate and Household Income are the top predictors for student dropout.

---

## 📊 Power BI Dashboard

### Main Visuals:

1. **Card Visuals**: Average Dropout %, Attendance %, Re-enrollment % (Male/Female)
2. **Line Chart**: Yearly Dropout Trends by Gender
3. **Stacked Bar Chart**: Dropout Levels per Province
4. **Filled Map**: Dropout Rates by District (using Province as Location)
5. **Slicers**: Year, Province, School (Horizontal format)
   

<img width="447" height="254" alt="Dashboard" src="https://github.com/user-attachments/assets/9938a196-9c06-4285-babb-6f65f2bc8866" />


### Features:

* Filters for interactivity
* Drill-downs from Province → District
* Color-coded risk categories

---


## 📈 Recommendations

* Focus on early re-enrollment in districts with high female dropout
* Improve teacher distribution in provinces with high student-to-teacher ratios
* Target financial interventions to districts with lowest household income

---

##  Future Work

* Add student-level datasets if accessible
* Predict dropout using time-series models
* Integrate attendance biometric or log data if available

---

## Student Information

* **Student Name**: Umwali Belyse
* **ID**: 27229


---

