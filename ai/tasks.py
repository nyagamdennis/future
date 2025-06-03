# new_project/your_app/tasks.py

# from celery import shared_task
from prophet import Prophet
import pandas as pd
import holidays
import time
import requests
import os

ai_url = 'https://murildennis.pythonanywhere.com/sales_ai/pass-data/'
def make_holidays_df(years):
    ke_holidays = holidays.Kenya(years=years)
    return pd.DataFrame([
        {"ds": pd.to_datetime(date), "holiday": name}
        for date, name in ke_holidays.items()
    ])



# @shared_task
def retrain_prophet_model(customer_id, sales_data):
    print('training session')
    # request_url = os.getenv("AI_URL")
    request_url = ai_url

    try:
        if len(sales_data) < 3:
            print(f"Not enough data for customer {customer_id}")
            return

        # Prepare sales data
        df = pd.DataFrame(sales_data)
        df.rename(columns={"date_sold": "ds", "total_amount": "y"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

        # Add Kenyan holidays
        years = list(set(df["ds"].dt.year)) + [pd.Timestamp.today().year]
        holidays_df = make_holidays_df(years)

        # Build & fit Prophet model
        model = Prophet(
            holidays=holidays_df,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1
        )
        model.add_seasonality(name='daily', period=1, fourier_order=3)  # Optional fine-tuning
        start = time.time()
        model.fit(df)
        print(f"Model training took {time.time() - start:.2f} seconds")

        # Forecast for 30 days ahead
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Pick realistic next date
        today = pd.Timestamp.today().normalize()
        hist_avg = df["y"].mean()
        threshold = hist_avg * 0.5

        future_preds = forecast[forecast["ds"] > today]
        candidates = future_preds[future_preds["yhat"] >= threshold]

        if not candidates.empty:
            next_date = candidates["ds"].iloc[0]
        else:
            next_date = future_preds.loc[future_preds["yhat"].idxmax(), "ds"]

        # Send prediction to your main app
        response = requests.post(
            request_url,
            json={
                "customer_id": customer_id,
                "predicted_date": next_date.isoformat()
            },
            timeout=10
        )
        print(f"Prediction sent for {customer_id}: {next_date} (status {response.status_code})")

    except Exception as e:
        print(f"Error while training for customer {customer_id}: {e}")


# @shared_task
# def retrain_prophet_model(customer_id, sales_data):
#     print('Training models')
#     request_url = os.getenv("AI_URL")  # ✅ Corrected

#     try:
#         if len(sales_data) < 3:
#             print(f'Not enough data for customer {customer_id}')
#             return

#         df = pd.DataFrame(sales_data)
#         df.rename(columns={"date_sold": "ds", "total_amount": "y"}, inplace=True)
#         df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

#         model = Prophet()
#         model.fit(df)

#         future = model.make_future_dataframe(periods=30)
#         forecast = model.predict(future)
#         next_date = forecast["ds"].iloc[-1]

#         # Send prediction back to main app
#         response = requests.post(
#             request_url,
#             json={
#                 "customer_id": customer_id,
#                 "predicted_date": next_date.isoformat()
#             },
#             timeout=10
#         )
#         print("Prediction sent:", response.status_code)

#     except Exception as e:
#         print(f'Error while training for customer {customer_id}: {e}')
