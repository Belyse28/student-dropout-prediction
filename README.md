# student-dropout-prediction

# 📘 Rwanda Student Dropout Prediction

---

## 🧭 Sector Selection

* **Selected Sector:** ✅ Education

---

##  Problem Statement

**Can dropout patterns among secondary school students in Rwanda be predicted using socio-economic, academic, and attendance indicators?**

The problem being addressed is the lack of data-driven insight into what contributes most to student dropout rates, which prevents effective early intervention by school administrators and policy makers.

---

## Dataset Identification

* **Dataset Title:** Rwanda Secondary School Education Statistics
* **Source:** Rwanda Open Data Portal / NISR
* **Number of Rows and Columns:** \~1,000 rows × 16 columns
* **Data Structure:** ✅ Structured (CSV)
* **Data Status:** ✅ Clean (after preprocessing)

---

## 🐍 Python Analytics Tasks

###  Clean the Dataset

```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("processed_dropout_data.csv")

# Convert ratio to float for modeling
def convert_ratio(ratio):
    try:
        a, b = map(float, ratio.split(':'))
        return a / b
    except:
        return np.nan

# Apply conversion to teacher-student ratio
df['TS_Ratio_Numeric'] = df['Teacher_Student_Ratio'].apply(convert_ratio)

# Drop missing important values
df = df.dropna(subset=['Overall_Dropout_Percentage', 'Year'])
```

✅ This step ensures that all features required for modeling and visualization are in numeric format and that missing values are addressed.

---

### 2️⃣ Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['Overall_Dropout_Percentage'], kde=True, bins=20)
plt.title('Distribution of Overall Dropout Percentage')
plt.xlabel('Dropout %')
plt.ylabel('Count')
plt.show()
```

🔍 A visual inspection shows dropout rates are generally skewed toward lower percentages but with notable spikes in some districts.

---

### 3️⃣ Machine Learning Model (Random Forest)

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

✅ The classifier gave accurate predictions for dropout level using key numeric indicators.

---

### 4️⃣ Innovation: Feature Importance

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

### Features:

* Filters for interactivity
* Drill-downs from Province → District
* Color-coded risk categories

---

## 🗂️ GitHub Repository Structure

```plaintext
📁 rwanda-dropout-prediction/
├── 📁 data/
│   └── processed_dropout_data.csv
├── 📁 notebooks/
│   └── dropout_modeling.ipynb
├── 📁 powerbi/
│   └── dropout_dashboard.pbix
├── 📁 presentation/
│   └── dropout_slides.pptx
├── README.md
└── requirements.txt
```

---

## ✅ Submission Summary

* 📌 Python notebook: ✔️ Cleaned and modeled data
* 📌 Power BI dashboard: ✔️ With slicers, cards, trend charts, maps
* 📌 GitHub repo: ✔️ With full structure and README
* 📌 Presentation: ✔️ Summarizes findings, methodology, and recommendations

---

## 📈 Recommendations

* Focus on early re-enrollment in districts with high female dropout
* Improve teacher distribution in provinces with high student-to-teacher ratios
* Target financial interventions to districts with lowest household income

---

## 📍 Future Work

* Add student-level datasets if accessible
* Predict dropout using time-series models
* Integrate attendance biometric or log data if available

---

## 🙋 Contact

* **Student Name**: Umwali Belyse
* **Course**: INSY 8413 | Introduction to Big Data Analytics
* **Instructor**: Eric Maniraguha
* **University**: AUCA, Faculty of Information Technology

---

> "Whatever you do, work at it with all your heart, as working for the Lord, not for human masters." — Colossians 3:23
