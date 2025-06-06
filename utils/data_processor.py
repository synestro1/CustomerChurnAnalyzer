import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_and_process_data():
    """
    Load and process the MTN customer churn data
    """
    try:
        # Load the data
        df = pd.read_csv("attached_assets/mtn_customer_churn.csv", encoding="utf-8")

        # Remove any BOM characters
        df.columns = df.columns.str.replace("﻿", "")

        # Basic data cleaning
        # Convert date column
        if "Date of Purchase" in df.columns:
            df["Date of Purchase"] = pd.to_datetime(
                df["Date of Purchase"], format="%b-%y", errors="coerce"
            )

        # Clean numeric columns
        numeric_columns = [
            "Age",
            "Satisfaction Rate",
            "Customer Tenure in months",
            "Unit Price",
            "Number of Times Purchased",
            "Total Revenue",
            "Data Usage",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle missing values
        # Fill missing churn reasons for non-churned customers
        df["Reasons for Churn"] = df["Reasons for Churn"].fillna("Not Churned")

        # Fill missing categorical values
        categorical_columns = [
            "State",
            "MTN Device",
            "Gender",
            "Customer Review",
            "Subscription Plan",
        ]
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        # Create derived features
        if "Total Revenue" in df.columns and "Customer Tenure in months" in df.columns:
            df["Revenue per Month"] = df["Total Revenue"] / df[
                "Customer Tenure in months"
            ].replace(0, 1)

        if "Data Usage" in df.columns and "Customer Tenure in months" in df.columns:
            df["Data Usage per Month"] = df["Data Usage"] / df[
                "Customer Tenure in months"
            ].replace(0, 1)

        # Create age groups
        if "Age" in df.columns:
            df["Age Group"] = pd.cut(
                df["Age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "55+"],
            )

        # Create tenure groups
        if "Customer Tenure in months" in df.columns:
            df["Tenure Group"] = pd.cut(
                df["Customer Tenure in months"],
                bins=[0, 6, 12, 24, 36, 100],
                labels=[
                    "0-6 months",
                    "7-12 months",
                    "13-24 months",
                    "25-36 months",
                    "36+ months",
                ],
            )

        return df

    except FileNotFoundError:
        st.error(
            "Data file not found. Please ensure 'attached_assets/mtn_customer_churn.csv' exists."
        )
        return None
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None


def get_summary_stats(df):
    """
    Generate summary statistics for the dataset
    """
    if df is None or df.empty:
        return None

    try:
        stats = {
            "total_customers": df["Customer ID"].nunique(),
            "total_records": len(df),
            "churned_customers": df[df["Customer Churn Status"] == "Yes"][
                "Customer ID"
            ].nunique(),
            "churn_rate": (
                df[df["Customer Churn Status"] == "Yes"]["Customer ID"].nunique()
                / df["Customer ID"].nunique()
            )
            * 100,
            "total_revenue": df["Total Revenue"].sum(),
            "avg_age": df["Age"].mean(),
            "avg_tenure": df["Customer Tenure in months"].mean(),
            "avg_satisfaction": df["Satisfaction Rate"].mean(),
            "unique_states": df["State"].nunique(),
            "unique_devices": df["MTN Device"].nunique(),
            "unique_plans": df["Subscription Plan"].nunique(),
        }
        return stats
    except Exception as e:
        st.error(f"Error calculating summary statistics: {str(e)}")
        return None


def get_churn_insights(df):
    """
    Generate key insights about customer churn
    """
    if df is None or df.empty:
        return None

    try:
        insights = {}

        # Basic churn metrics
        total_customers = df["Customer ID"].nunique()
        churned_customers = df[df["Customer Churn Status"] == "Yes"][
            "Customer ID"
        ].nunique()
        insights["churn_rate"] = (churned_customers / total_customers) * 100

        # Satisfaction analysis
        churned_satisfaction = df[df["Customer Churn Status"] == "Yes"][
            "Satisfaction Rate"
        ].mean()
        retained_satisfaction = df[df["Customer Churn Status"] == "No"][
            "Satisfaction Rate"
        ].mean()
        insights["avg_satisfaction_churned"] = churned_satisfaction
        insights["avg_satisfaction_retained"] = retained_satisfaction

        # Top churn reasons
        churn_reasons = df[df["Customer Churn Status"] == "Yes"][
            "Reasons for Churn"
        ].value_counts()
        insights["top_churn_reason"] = (
            churn_reasons.index[0] if len(churn_reasons) > 0 else "Unknown"
        )
        insights["churn_reasons_distribution"] = churn_reasons.to_dict()

        # State analysis
        state_churn = df.groupby("State").agg(
            {
                "Customer ID": "nunique",
                "Customer Churn Status": lambda x: (x == "Yes").sum(),
            }
        )
        state_churn["churn_rate"] = (
            state_churn["Customer Churn Status"] / state_churn["Customer ID"]
        ) * 100
        high_churn_states = state_churn[
            state_churn["churn_rate"] > insights["churn_rate"]
        ].index.tolist()
        insights["high_churn_states"] = high_churn_states

        # Device analysis
        device_churn = df.groupby("MTN Device").agg(
            {
                "Customer ID": "nunique",
                "Customer Churn Status": lambda x: (x == "Yes").sum(),
            }
        )
        device_churn["churn_rate"] = (
            device_churn["Customer Churn Status"] / device_churn["Customer ID"]
        ) * 100
        insights["device_churn_rates"] = device_churn["churn_rate"].to_dict()

        # Revenue impact
        churned_revenue = df[df["Customer Churn Status"] == "Yes"][
            "Total Revenue"
        ].sum()
        total_revenue = df["Total Revenue"].sum()
        insights["revenue_lost_to_churn"] = churned_revenue
        insights["revenue_loss_percentage"] = (churned_revenue / total_revenue) * 100

        # Age and tenure insights
        churned_age = df[df["Customer Churn Status"] == "Yes"]["Age"].mean()
        retained_age = df[df["Customer Churn Status"] == "No"]["Age"].mean()
        insights["avg_age_churned"] = churned_age
        insights["avg_age_retained"] = retained_age

        churned_tenure = df[df["Customer Churn Status"] == "Yes"][
            "Customer Tenure in months"
        ].mean()
        retained_tenure = df[df["Customer Churn Status"] == "No"][
            "Customer Tenure in months"
        ].mean()
        insights["avg_tenure_churned"] = churned_tenure
        insights["avg_tenure_retained"] = retained_tenure

        return insights

    except Exception as e:
        st.error(f"Error generating churn insights: {str(e)}")
        return None


def filter_data(df, filters):
    """
    Apply filters to the dataset
    """
    if df is None or df.empty:
        return None

    try:
        filtered_df = df.copy()

        # Apply each filter
        for column, values in filters.items():
            if column in df.columns and values:
                if isinstance(values, list):
                    filtered_df = filtered_df[filtered_df[column].isin(values)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == values]

        return filtered_df

    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return None


def get_customer_profile(df, customer_id):
    """
    Get detailed profile for a specific customer
    """
    if df is None or df.empty:
        return None

    try:
        customer_data = df[df["Customer ID"] == customer_id]

        if customer_data.empty:
            return None

        # Aggregate customer information
        profile = {
            "customer_id": customer_id,
            "full_name": customer_data["Full Name"].iloc[0],
            "age": customer_data["Age"].iloc[0],
            "gender": customer_data["Gender"].iloc[0],
            "state": customer_data["State"].iloc[0],
            "churn_status": customer_data["Customer Churn Status"].iloc[0],
            "satisfaction_rate": customer_data["Satisfaction Rate"].mean(),
            "total_revenue": customer_data["Total Revenue"].sum(),
            "avg_tenure": customer_data["Customer Tenure in months"].mean(),
            "devices_used": customer_data["MTN Device"].unique().tolist(),
            "subscription_plans": customer_data["Subscription Plan"].unique().tolist(),
            "total_data_usage": customer_data["Data Usage"].sum(),
            "number_of_purchases": customer_data["Number of Times Purchased"].sum(),
        }

        if customer_data["Customer Churn Status"].iloc[0] == "Yes":
            churn_reasons = customer_data["Reasons for Churn"].unique()
            profile["churn_reasons"] = [
                reason for reason in churn_reasons if reason and reason != "Not Churned"
            ]

        return profile

    except Exception as e:
        st.error(f"Error getting customer profile: {str(e)}")
        return None
