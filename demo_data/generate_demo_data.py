"""Generate sample demo datasets for insight-engine."""

import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_ecommerce():
    """E-commerce transactions dataset."""
    categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports", "Beauty"]
    regions = ["West", "East", "Midwest", "South", "Northeast"]
    channels = ["Organic Search", "Paid Ads", "Social Media", "Email", "Direct", "Referral"]
    payment_methods = ["Credit Card", "PayPal", "Debit Card", "Apple Pay"]

    rows = []
    start_date = datetime(2024, 1, 1)

    for i in range(1000):
        date = start_date + timedelta(days=random.randint(0, 364))
        category = random.choice(categories)
        base_price = {"Electronics": 150, "Clothing": 45, "Home & Garden": 75, "Books": 18, "Sports": 60, "Beauty": 35}[
            category
        ]
        quantity = random.randint(1, 5)
        unit_price = round(base_price * random.uniform(0.5, 2.5), 2)
        discount = round(random.uniform(0, 0.3), 2) if random.random() > 0.6 else 0
        revenue = round(unit_price * quantity * (1 - discount), 2)
        returned = 1 if random.random() < 0.08 else 0

        rows.append(
            {
                "order_id": f"ORD-{10000 + i}",
                "date": date.strftime("%Y-%m-%d"),
                "customer_id": f"CUST-{random.randint(1, 300)}",
                "category": category,
                "product_name": f"{category} Item {random.randint(1, 50)}",
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_pct": discount,
                "revenue": revenue,
                "region": random.choice(regions),
                "channel": random.choice(channels),
                "payment_method": random.choice(payment_methods),
                "returned": returned,
                "customer_rating": round(random.uniform(1, 5), 1) if random.random() > 0.3 else None,
            }
        )

    path = os.path.join(DEMO_DIR, "ecommerce.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_marketing_touchpoints():
    """Marketing touchpoints dataset for attribution modeling."""
    channels = ["Google Ads", "Facebook Ads", "Email", "Organic Search", "LinkedIn", "Webinar"]
    rows = []
    start_date = datetime(2024, 1, 1)

    for user_id in range(1, 201):
        n_touches = random.randint(1, 8)
        base_date = start_date + timedelta(days=random.randint(0, 300))
        converted = random.random() < 0.35

        for j in range(n_touches):
            touch_date = base_date + timedelta(days=random.randint(0, 30) * j)
            rows.append(
                {
                    "user_id": f"U{user_id:04d}",
                    "channel": random.choice(channels),
                    "timestamp": touch_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "campaign": f"Campaign_{random.choice(['Spring', 'Summer', 'Fall', 'Winter'])}_{random.randint(1, 5)}",
                    "spend": round(random.uniform(0.5, 25.0), 2),
                    "converted": 1 if converted and j == n_touches - 1 else 0,
                }
            )

    path = os.path.join(DEMO_DIR, "marketing_touchpoints.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_hr_attrition():
    """HR attrition dataset for predictive modeling."""
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    performance_ratings = [1, 2, 3, 4, 5]
    rows = []

    for i in range(500):
        dept = random.choice(departments)
        years = random.randint(0, 25)
        salary = round(
            random.gauss(
                {
                    "Engineering": 95000,
                    "Sales": 70000,
                    "Marketing": 65000,
                    "HR": 60000,
                    "Finance": 80000,
                    "Operations": 55000,
                }[dept],
                15000,
            )
        )
        performance = random.choice(performance_ratings)
        satisfaction = round(random.uniform(1, 5), 1)
        overtime = 1 if random.random() < 0.3 else 0

        # Attrition more likely with low satisfaction, low salary, high overtime
        attrition_prob = 0.05
        if satisfaction < 2.5:
            attrition_prob += 0.2
        if salary < 50000:
            attrition_prob += 0.15
        if overtime:
            attrition_prob += 0.1
        if years < 2:
            attrition_prob += 0.1

        attrited = 1 if random.random() < attrition_prob else 0

        rows.append(
            {
                "employee_id": f"EMP-{1000 + i}",
                "department": dept,
                "education": random.choice(education_levels),
                "years_at_company": years,
                "monthly_salary": salary,
                "performance_rating": performance,
                "job_satisfaction": satisfaction,
                "overtime": overtime,
                "distance_from_home_miles": random.randint(1, 50),
                "num_companies_worked": random.randint(0, 8),
                "training_hours_last_year": random.randint(0, 100),
                "attrition": attrited,
            }
        )

    path = os.path.join(DEMO_DIR, "hr_attrition.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_time_series_sales():
    """Time series sales dataset for forecasting."""
    import math

    rows = []
    start_date = datetime(2024, 1, 1)

    for day in range(365):
        date = start_date + timedelta(days=day)
        trend = 100 + 0.1 * day
        seasonality = 20 * math.sin(2 * math.pi * day / 90)
        noise = random.uniform(-10, 10)
        sales = round(trend + seasonality + noise, 2)

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "sales": sales,
                "trend": round(trend, 2),
                "seasonality": round(seasonality, 2),
            }
        )

    path = os.path.join(DEMO_DIR, "time_series_sales.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    print(f"Generated: {generate_ecommerce()}")
    print(f"Generated: {generate_marketing_touchpoints()}")
    print(f"Generated: {generate_hr_attrition()}")
    print(f"Generated: {generate_time_series_sales()}")
