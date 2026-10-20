# COVID-19 Trend Analysis & 7-Day Forecasting with Python

**IntelliPaat Data Science Training Project**

---

## 📌 Project Overview & Problem Statement
Given data about COVID-19 patients, this project visualizes the impact, analyzes the trends of infection and recovery rates across the globe and in India, and makes predictions about the number of cases expected **one week into the future** based on current trends.

---

## 🗂️ Project Directory Structure

```
covid19_trend_analysis/
│
├── data/
│   └── Covid_19_Clean_Complete.csv           # Cleaned COVID-19 dataset (Johns Hopkins / Kaggle)
│
├── outputs/                                  # Exported interactive visualizations & predictions
│   ├── 01_worldwide_trends.html              # Global cumulative trajectory
│   ├── 02_worldwide_daily_rates.html         # Daily infection rate vs recovery rate
│   ├── 03_top10_countries.html               # Top 10 most affected nations
│   ├── 04_india_trends.html                  # India cumulative progression
│   ├── 05_india_daily_rates.html             # India daily infection & recovery dynamics
│   ├── 06_worldwide_prophet_forecast.html    # Prophet 7-day forecast for Worldwide cases
│   ├── 07_india_prophet_forecast.html        # Prophet 7-day forecast for India cases
│   └── 7_day_predictions_summary.csv         # Tabular 7-day predictions table
│
├── covid_19_analysis.ipynb                  # Fully executed Jupyter Notebook with charts & markdown
├── covid_19_analysis.py                     # Standalone production-ready Python script
└── README.md                                 # Project documentation and comparison guide
```

---

## 🚀 How to Run the Project

### Option 1: Run the Standalone Script
Open PowerShell or your command prompt in this directory and execute:
```bash
python covid_19_analysis.py
```
This will:
- Process the dataset.
- Calculate all global and India metrics.
- Train Facebook Prophet models for Worldwide and India.
- Save all interactive HTML charts and CSV forecast summaries in `outputs/`.

### Option 2: Open and Run the Jupyter Notebook
Open VS Code, JupyterLab, or Jupyter Notebook:
```bash
jupyter notebook covid_19_analysis.ipynb
```
All cells are already executed, complete with inline interactive Plotly graphs, tables, and step-by-step explanations.

---

## 📊 Step-by-Step Methodology

### 1. Data Ingestion & Preprocessing (Pandas)
- Loaded `Covid_19_Clean_Complete.csv`.
- Converted `Date` to `datetime64[ns]` to enable chronological time-series slicing.
- Handled missing values (`Province/State` populated with `'All'`).
- Sorted chronologically.

### 2. Epidemiological Metric Formulas
- **Daily New Cases / Infection Rate**:
  $$\text{Daily\_Confirmed}_t = \text{Confirmed}_t - \text{Confirmed}_{t-1}$$
- **Daily Recoveries**:
  $$\text{Daily\_Recovered}_t = \text{Recovered}_t - \text{Recovered}_{t-1}$$
- **Mortality Rate (%)**:
  $$\text{Mortality Rate} = \left(\frac{\text{Cumulative Deaths}}{\text{Cumulative Confirmed}}\right) \times 100$$
- **Recovery Rate (%)**:
  $$\text{Recovery Rate} = \left(\frac{\text{Cumulative Recovered}}{\text{Cumulative Confirmed}}\right) \times 100$$
- **7-Day Rolling Average**:
  $$\text{MA}_{7}(t) = \frac{1}{7} \sum_{k=0}^{6} x_{t-k}$$

### 3. Interactive Data Visualizations (Plotly)
- **Cumulative Trajectories**: Compares active, recovered, deaths, and confirmed cases dynamically.
- **Daily Rates**: Overlay bars with rolling averages to clearly spot infection peaks vs recovery curves.
- **Top 10 Countries**: Ranked horizontal bar chart identifying epicenters (US, Brazil, India, etc.).

### 4. Time Series Modeling & 7-Day Forecasting (Facebook Prophet)
- **Model formulation**: $y(t) = g(t) + s(t) + h(t) + \epsilon_t$
- Required schema: `ds` (date) and `y` (target metric).
- Configured with `interval_width=0.95` (95% uncertainty interval), `weekly_seasonality=True`, and `daily_seasonality=False`.
- Generated predictions for the next 7 days (`periods=7`).

---

## 🎯 7-Day Future Predictions Summary

| Date | Day | Worldwide Expected Cases | Worldwide 95% Interval | India Expected Cases | India 95% Interval |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2020-07-28** | Tuesday | **16,324,014** | [16.17M - 16.49M] | **1,352,019** | [1.30M - 1.40M] |
| **2020-07-29** | Wednesday | **16,534,051** | [16.37M - 16.69M] | **1,379,016** | [1.33M - 1.43M] |
| **2020-07-30** | Thursday | **16,748,199** | [16.58M - 16.90M] | **1,406,215** | [1.35M - 1.46M] |
| **2020-07-31** | Friday | **16,963,623** | [16.80M - 17.13M] | **1,433,542** | [1.38M - 1.49M] |
| **2020-08-01** | Saturday | **17,171,431** | [17.00M - 17.34M] | **1,460,594** | [1.41M - 1.52M] |
| **2020-08-02** | Sunday | **17,369,220** | [17.21M - 17.55M] | **1,487,789** | [1.44M - 1.54M] |
| **2020-08-03** | Monday | **17,564,098** | [17.37M - 17.76M] | **1,514,358** | [1.46M - 1.57M] |

---

## 🔍 Common Mistakes Checklist (Compare Your Code With This!)

If you were stuck or noticed differences in your project, check these common pitfalls:

### 1. Feeding Non-Aggregated Data to Prophet
- **Mistake**: Passing the raw dataframe directly into Prophet. Raw data has multiple rows per date (one for each province or country). Prophet requires exactly **one observation per timestamp**.
- **Fix**: Always group by date first:
  ```python
  world_daily = df.groupby('Date')['Confirmed'].sum().reset_index()
  ```

### 2. Incorrect Column Names for Prophet
- **Mistake**: Using column names like `'Date'` and `'Confirmed'` when calling `model.fit()`.
- **Fix**: Prophet strictly expects columns named `'ds'` and `'y'`:
  ```python
  prophet_df = world_daily.rename(columns={'Date': 'ds', 'Confirmed': 'y'})
  ```

### 3. Missing Datetime Conversion
- **Mistake**: Leaving `'Date'` as an `object` (string) type.
- **Fix**: Convert it explicitly:
  ```python
  df['Date'] = pd.to_datetime(df['Date'])
  ```

### 4. Negative Values from `.diff()`
- **Mistake**: Computing `.diff()` without clipping. In real-world COVID datasets, occasional government data revisions cause negative differences.
- **Fix**: Use `.clip(lower=0)`:
  ```python
  daily_confirmed = confirmed_series.diff().fillna(0).clip(lower=0)
  ```

### 5. Enabling `daily_seasonality` for Daily Data
- **Mistake**: Setting `daily_seasonality=True` in Prophet when data is recorded once per day.
- **Fix**: `daily_seasonality` is for intraday/hourly data. For daily data, use `daily_seasonality=False` and `weekly_seasonality=True` (to capture weekend reporting drops).

### 6. Cumulative vs. Daily Confusion
- The dataset gives **cumulative** counts (total cases up to that date).
- If predicting cumulative confirmed cases, fit Prophet directly on cumulative totals.
- If predicting daily new cases, fit Prophet on daily increment numbers. Both are valid, but keep interpretations consistent!