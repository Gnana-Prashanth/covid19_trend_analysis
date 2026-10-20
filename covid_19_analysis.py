"""
================================================================================
IntelliPaat Project: Analyzing the Trends of COVID-19 with Python
================================================================================
Problem Statement:
Given data about COVID-19 patients, write code to visualize the impact and 
analyze the trend of rate of infection and recovery as well as make predictions 
about the number of cases expected a week in future based on the current trends.

Guidelines:
- Use pandas to accumulate data from multiple perspectives / data aggregations.
- Use plotly (visualization library) to create interactive visualizations.
- Use Facebook prophet library to make time series models.
- Visualize the prediction by combining these technologies.

Author / Developer: Data Science Pair Programmer
Dataset: Covid_19_Clean_Complete.csv
================================================================================
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet


# ------------------------------------------------------------------------------
# STEP 1: DIRECTORY & FILE PATH SETUP
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Covid_19_Clean_Complete.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw COVID-19 CSV data and cleans it:
    1. Converts 'Date' column to standard pandas datetime.
    2. Fills missing 'Province/State' entries with 'All'.
    3. Validates essential columns: Confirmed, Deaths, Recovered, Active.
    """
    print("=" * 70)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[INFO] Raw Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 1. Convert Date to Datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Fill missing Province/State values
    df['Province/State'] = df['Province/State'].fillna('All')
    
    # 3. Sort chronologically
    df = df.sort_values(by=['Date', 'Country/Region']).reset_index(drop=True)
    
    print(f"[INFO] Date range: from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"[INFO] Total Unique Countries: {df['Country/Region'].nunique()}")
    
    return df


def aggregate_worldwide_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates country-level data across the globe by Date.
    Calculates:
    - Cumulative: Confirmed, Deaths, Recovered, Active
    - Daily changes: Daily_Confirmed (Infection Rate), Daily_Recovered, Daily_Deaths
    - Epidemiological Rates: Mortality_Rate (%), Recovery_Rate (%)
    """
    print("\n" + "=" * 70)
    print("STEP 2: AGGREGATING WORLDWIDE METRICS & RATES")
    print("=" * 70)
    
    world_df = df.groupby('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
    
    # Calculate daily new counts using .diff()
    # (First day diff will be NaN, fill with initial value)
    world_df['Daily_Confirmed'] = world_df['Confirmed'].diff().fillna(world_df['Confirmed'].iloc[0])
    world_df['Daily_Recovered'] = world_df['Recovered'].diff().fillna(world_df['Recovered'].iloc[0])
    world_df['Daily_Deaths'] = world_df['Deaths'].diff().fillna(world_df['Deaths'].iloc[0])
    
    # Ensure no negative diff values due to data reporting corrections
    world_df['Daily_Confirmed'] = world_df['Daily_Confirmed'].clip(lower=0)
    world_df['Daily_Recovered'] = world_df['Daily_Recovered'].clip(lower=0)
    world_df['Daily_Deaths'] = world_df['Daily_Deaths'].clip(lower=0)
    
    # Calculate key epidemiological rates
    world_df['Mortality_Rate_%'] = (world_df['Deaths'] / world_df['Confirmed']) * 100
    world_df['Recovery_Rate_%'] = (world_df['Recovered'] / world_df['Confirmed']) * 100
    
    # 7-day rolling average of daily cases to smooth reporting spikes
    world_df['Daily_Confirmed_7D_Avg'] = world_df['Daily_Confirmed'].rolling(window=7, min_periods=1).mean()
    world_df['Daily_Recovered_7D_Avg'] = world_df['Daily_Recovered'].rolling(window=7, min_periods=1).mean()
    
    latest = world_df.iloc[-1]
    print(f"[SUMMARY] Worldwide Total as of {latest['Date'].strftime('%Y-%m-%d')}:")
    print(f"  - Confirmed Cases: {int(latest['Confirmed']):,}")
    print(f"  - Recovered:       {int(latest['Recovered']):,} ({latest['Recovery_Rate_%']:.2f}%)")
    print(f"  - Deaths:          {int(latest['Deaths']):,} ({latest['Mortality_Rate_%']:.2f}%)")
    print(f"  - Active Cases:    {int(latest['Active']):,}")
    
    return world_df


def aggregate_india_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and aggregates India-specific records.
    Calculates daily infection rate, recovery rate, and mortality rate for India.
    """
    print("\n" + "=" * 70)
    print("STEP 3: AGGREGATING INDIA METRICS & RATES")
    print("=" * 70)
    
    india_df = df[df['Country/Region'] == 'India'].groupby('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
    
    # Daily changes
    india_df['Daily_Confirmed'] = india_df['Confirmed'].diff().fillna(india_df['Confirmed'].iloc[0]).clip(lower=0)
    india_df['Daily_Recovered'] = india_df['Recovered'].diff().fillna(india_df['Recovered'].iloc[0]).clip(lower=0)
    india_df['Daily_Deaths'] = india_df['Deaths'].diff().fillna(india_df['Deaths'].iloc[0]).clip(lower=0)
    
    # Rates
    india_df['Mortality_Rate_%'] = (india_df['Deaths'] / india_df['Confirmed']) * 100
    india_df['Recovery_Rate_%'] = (india_df['Recovered'] / india_df['Confirmed']) * 100
    
    # 7-day rolling average
    india_df['Daily_Confirmed_7D_Avg'] = india_df['Daily_Confirmed'].rolling(window=7, min_periods=1).mean()
    india_df['Daily_Recovered_7D_Avg'] = india_df['Daily_Recovered'].rolling(window=7, min_periods=1).mean()
    
    latest_ind = india_df.iloc[-1]
    print(f"[SUMMARY] India Total as of {latest_ind['Date'].strftime('%Y-%m-%d')}:")
    print(f"  - Confirmed Cases: {int(latest_ind['Confirmed']):,}")
    print(f"  - Recovered:       {int(latest_ind['Recovered']):,} ({latest_ind['Recovery_Rate_%']:.2f}%)")
    print(f"  - Deaths:          {int(latest_ind['Deaths']):,} ({latest_ind['Mortality_Rate_%']:.2f}%)")
    print(f"  - Active Cases:    {int(latest_ind['Active']):,}")
    
    return india_df


def create_interactive_visualizations(df: pd.DataFrame, world_df: pd.DataFrame, india_df: pd.DataFrame):
    """
    Generates rich, interactive Plotly visualizations:
    1. Global Cumulative Trends (Confirmed, Recovered, Deaths, Active)
    2. Global Infection & Recovery Dynamics (Daily counts + Rates)
    3. Top 10 Most Impacted Countries
    4. India Progression Trends
    5. India Daily New Cases vs Recoveries
    All figures are saved to the outputs/ directory as interactive HTML files.
    """
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING INTERACTIVE PLOTLY VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Worldwide Cumulative Trends
    fig_world_cum = go.Figure()
    fig_world_cum.add_trace(go.Scatter(x=world_df['Date'], y=world_df['Confirmed'], mode='lines', name='Confirmed', line=dict(color='#ff7f0e', width=3)))
    fig_world_cum.add_trace(go.Scatter(x=world_df['Date'], y=world_df['Recovered'], mode='lines', name='Recovered', line=dict(color='#2ca02c', width=3)))
    fig_world_cum.add_trace(go.Scatter(x=world_df['Date'], y=world_df['Active'], mode='lines', name='Active', line=dict(color='#1f77b4', width=2, dash='dash')))
    fig_world_cum.add_trace(go.Scatter(x=world_df['Date'], y=world_df['Deaths'], mode='lines', name='Deaths', line=dict(color='#d62728', width=2)))
    
    fig_world_cum.update_layout(
        title='<b>COVID-19 Global Cumulative Trajectory (Worldwide)</b>',
        xaxis_title='Date',
        yaxis_title='Total Count of Cases',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    p1 = os.path.join(OUTPUT_DIR, "01_worldwide_trends.html")
    fig_world_cum.write_html(p1)
    print(f"[SAVED] {p1}")

    # 2. Worldwide Daily Infection & Recovery Rates
    fig_world_daily = go.Figure()
    fig_world_daily.add_trace(go.Bar(x=world_df['Date'], y=world_df['Daily_Confirmed'], name='Daily New Cases', marker_color='rgba(255, 127, 14, 0.6)'))
    fig_world_daily.add_trace(go.Bar(x=world_df['Date'], y=world_df['Daily_Recovered'], name='Daily Recovered', marker_color='rgba(44, 160, 44, 0.6)'))
    fig_world_daily.add_trace(go.Scatter(x=world_df['Date'], y=world_df['Daily_Confirmed_7D_Avg'], mode='lines', name='New Cases (7D Avg)', line=dict(color='#d62728', width=2.5)))
    
    fig_world_daily.update_layout(
        title='<b>Global Daily Rate of Infection and Recovery</b>',
        xaxis_title='Date',
        yaxis_title='Daily Count',
        barmode='overlay',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    p2 = os.path.join(OUTPUT_DIR, "02_worldwide_daily_rates.html")
    fig_world_daily.write_html(p2)
    print(f"[SAVED] {p2}")

    # 3. Top 10 Most Affected Countries
    latest_date = df['Date'].max()
    latest_country = df[df['Date'] == latest_date].groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
    top10_countries = latest_country.sort_values(by='Confirmed', ascending=False).head(10)
    
    fig_top10 = px.bar(
        top10_countries,
        x='Confirmed',
        y='Country/Region',
        orientation='h',
        color='Confirmed',
        color_continuous_scale='Reds',
        text='Confirmed',
        title=f'<b>Top 10 Most Affected Countries (as of {latest_date.strftime("%b %d, %Y")})</b>'
    )
    fig_top10.update_layout(
        yaxis=dict(autorange="reversed"),
        template='plotly_white',
        xaxis_title='Total Confirmed Cases',
        yaxis_title='Country'
    )
    p3 = os.path.join(OUTPUT_DIR, "03_top10_countries.html")
    fig_top10.write_html(p3)
    print(f"[SAVED] {p3}")

    # 4. India Cumulative Trends
    fig_india_cum = go.Figure()
    fig_india_cum.add_trace(go.Scatter(x=india_df['Date'], y=india_df['Confirmed'], mode='lines+markers', name='Confirmed', line=dict(color='#ff7f0e', width=3)))
    fig_india_cum.add_trace(go.Scatter(x=india_df['Date'], y=india_df['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='#2ca02c', width=3)))
    fig_india_cum.add_trace(go.Scatter(x=india_df['Date'], y=india_df['Active'], mode='lines', name='Active', line=dict(color='#1f77b4', width=2, dash='dot')))
    fig_india_cum.add_trace(go.Scatter(x=india_df['Date'], y=india_df['Deaths'], mode='lines', name='Deaths', line=dict(color='#d62728', width=2)))
    
    fig_india_cum.update_layout(
        title='<b>COVID-19 Cumulative Trends in India</b>',
        xaxis_title='Date',
        yaxis_title='Total Count',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    p4 = os.path.join(OUTPUT_DIR, "04_india_trends.html")
    fig_india_cum.write_html(p4)
    print(f"[SAVED] {p4}")

    # 5. India Daily Rate of Infection and Recovery
    fig_india_daily = go.Figure()
    fig_india_daily.add_trace(go.Bar(x=india_df['Date'], y=india_df['Daily_Confirmed'], name='Daily New Cases', marker_color='rgba(255, 127, 14, 0.7)'))
    fig_india_daily.add_trace(go.Bar(x=india_df['Date'], y=india_df['Daily_Recovered'], name='Daily Recovered', marker_color='rgba(44, 160, 44, 0.7)'))
    fig_india_daily.add_trace(go.Scatter(x=india_df['Date'], y=india_df['Daily_Confirmed_7D_Avg'], mode='lines', name='7-Day Avg New Cases', line=dict(color='#d62728', width=2.5)))
    
    fig_india_daily.update_layout(
        title='<b>India Daily Infection Rate vs Recovery Rate</b>',
        xaxis_title='Date',
        yaxis_title='Daily Count of Patients',
        barmode='group',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    p5 = os.path.join(OUTPUT_DIR, "05_india_daily_rates.html")
    fig_india_daily.write_html(p5)
    print(f"[SAVED] {p5}")


def run_prophet_forecasting(series_df: pd.DataFrame, region_name: str, forecast_days: int = 7):
    """
    Builds and trains a Facebook Prophet time-series forecasting model.
    Predicts cases for the upcoming 'forecast_days' (1 week) into the future.
    """
    print(f"\n[PROPHET MODEL] Training Time-Series Model for {region_name}...")
    
    # 1. Format for Prophet: Requires exact column names 'ds' (datestamp) and 'y' (target)
    prophet_input = series_df[['Date', 'Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'}).copy()
    
    # 2. Fit Model
    model = Prophet(
        interval_width=0.95,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_input)
    
    # 3. Future DataFrame for 7 days
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    
    # Round predictions
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[col] = forecast[col].round().astype(int)
        
    print(f"[PROPHET MODEL] Forecasting complete for next {forecast_days} days.")
    
    # 4. Interactive Visualization
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(
        x=prophet_input['ds'],
        y=prophet_input['y'],
        mode='markers',
        name='Actual Observed Cases',
        marker=dict(color='black', size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Prophet Forecast (yhat)',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Confidence Interval Upper
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='95% Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Confidence Interval Lower with fill
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='95% Confidence Interval',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    last_actual_date = prophet_input['ds'].max()
    fig.add_vline(
        x=last_actual_date.timestamp() * 1000,
        line_dash="dash",
        line_color="red",
        annotation_text="Forecast Horizon (Next 7 Days)",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=f'<b>{region_name} COVID-19 Confirmed Cases: Prophet 7-Day Prediction</b>',
        xaxis_title='Date',
        yaxis_title='Confirmed Cases',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return model, forecast, fig


def main():
    print("=" * 70)
    print("STARTING COMPLETE COVID-19 TREND ANALYSIS & PREDICTION PIPELINE")
    print("=" * 70)
    
    # 1. Load data
    df = load_and_preprocess_data(DATA_PATH)
    
    # 2. Aggregations
    world_df = aggregate_worldwide_data(df)
    india_df = aggregate_india_data(df)
    
    # 3. Visualizations
    create_interactive_visualizations(df, world_df, india_df)
    
    # 4. Worldwide Prophet Forecasting (Next 7 Days)
    print("\n" + "=" * 70)
    print("STEP 5: TIME SERIES FORECASTING WITH FACEBOOK PROPHET (NEXT 7 DAYS)")
    print("=" * 70)
    
    _, world_forecast, world_fig = run_prophet_forecasting(world_df, region_name="Worldwide", forecast_days=7)
    p6 = os.path.join(OUTPUT_DIR, "06_worldwide_prophet_forecast.html")
    world_fig.write_html(p6)
    print(f"[SAVED] {p6}")
    
    # 5. India Prophet Forecasting (Next 7 Days)
    _, india_forecast, india_fig = run_prophet_forecasting(india_df, region_name="India", forecast_days=7)
    p7 = os.path.join(OUTPUT_DIR, "07_india_prophet_forecast.html")
    india_fig.write_html(p7)
    print(f"[SAVED] {p7}")
    
    # 6. Extract and Save 7-Day Forecast Summary Table
    future_world = world_forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    future_world.columns = ['Date', 'World_Predicted', 'World_Lower_95%', 'World_Upper_95%']
    
    future_india = india_forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    future_india.columns = ['Date', 'India_Predicted', 'India_Lower_95%', 'India_Upper_95%']
    
    summary_table = pd.merge(future_world, future_india, on='Date')
    summary_path = os.path.join(OUTPUT_DIR, "7_day_predictions_summary.csv")
    summary_table.to_csv(summary_path, index=False)
    print(f"[SAVED] {summary_path}")
    
    print("\n" + "=" * 70)
    print("7-DAY FUTURE PREDICTIONS (EXPECTED CASES ONE WEEK AHEAD):")
    print("=" * 70)
    for _, row in summary_table.iterrows():
        d_str = row['Date'].strftime('%Y-%m-%d (%A)')
        w_pred = f"{int(row['World_Predicted']):,}"
        i_pred = f"{int(row['India_Predicted']):,}"
        print(f"Date: {d_str} | World Expected: {w_pred:>12} | India Expected: {i_pred:>10}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All interactive visualizations and tables are ready in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()