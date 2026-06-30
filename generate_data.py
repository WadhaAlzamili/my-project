import pandas as pd
import random
from datetime import datetime, timedelta

patients = [101, 102, 103, 104, 105]

start_date = datetime(2026, 3, 1)
end_date = datetime(2026, 4, 30)

rows = []

current_date = start_date
while current_date <= end_date:
    for patient in patients:
        base_hr = random.randint(70, 95)
        base_bp = random.randint(110, 130)
        base_spo2 = random.randint(96, 99)
        base_temp = round(random.uniform(36.6, 37.4), 1)
        base_rr = random.randint(14, 18)

        # نخلي بعض المرضى يطلعون Warning/Critical أحيانًا
        if patient == 103 and random.random() < 0.35:
            base_hr = random.randint(112, 125)
            base_bp = random.randint(160, 175)
            base_spo2 = random.randint(91, 94)
            base_temp = round(random.uniform(38.0, 38.8), 1)
            base_rr = random.randint(20, 24)

        if patient == 105 and random.random() < 0.45:
            base_hr = random.randint(132, 145)
            base_bp = random.randint(182, 195)
            base_spo2 = random.randint(86, 90)
            base_temp = round(random.uniform(39.0, 39.8), 1)
            base_rr = random.randint(26, 30)

        timestamp = current_date.replace(hour=8, minute=0, second=0)

        last_med = current_date.replace(hour=6, minute=0, second=0)

        # بعض المرضى متأخرين وبعضهم قريب موعدهم
        if patient == 105:
            next_med = current_date.replace(hour=7, minute=20, second=0)
        elif patient == 103:
            next_med = current_date.replace(hour=8, minute=20, second=0)
        else:
            next_med = current_date.replace(hour=10, minute=0, second=0)

        rows.append([
            patient,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            base_hr,
            base_bp,
            base_spo2,
            base_temp,
            base_rr,
            last_med.strftime("%Y-%m-%d %H:%M:%S"),
            next_med.strftime("%Y-%m-%d %H:%M:%S")
        ])

    current_date += timedelta(days=1)

df = pd.DataFrame(rows, columns=[
    "Patient_ID",
    "Timestamp",
    "HR",
    "Systolic_BP",
    "SpO2",
    "Temp",
    "Resp_Rate",
    "Last_Med_Time",
    "Next_Med_Time"
])

df.to_csv("vitals_demo.csv", index=False)
print("vitals_demo.csv created successfully")
print(df.head())