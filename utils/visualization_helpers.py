import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_churn_distribution(df):
    """
    Create a comprehensive churn distribution visualization
    """
    # Overall churn distribution
    churn_counts = df["Customer Churn Status"].value_counts()

    # Create subplot with pie chart and bar chart
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Churn Distribution", "Churn by Customer Count"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
    )

    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            name="Churn Status",
            marker_colors=["#90EE90", "#FF6B6B"],
        ),
        row=1,
        col=1,
    )

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=churn_counts.index,
            y=churn_counts.values,
            name="Customer Count",
            marker_color=["#90EE90", "#FF6B6B"],
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Customer Churn Overview", showlegend=False, height=400
    )

    return fig


def create_age_analysis(df):
    """
    Create age distribution analysis by churn status
    """
    fig = px.histogram(
        df,
        x="Age",
        color="Customer Churn Status",
        barmode="overlay",
        title="Age Distribution by Churn Status",
        labels={"count": "Number of Customers"},
        color_discrete_map={"No": "#90EE90", "Yes": "#FF6B6B"},
        opacity=0.7,
        nbins=20,
    )

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Customers",
        legend_title="Churn Status",
    )

    return fig


def create_tenure_analysis(df):
    """
    Create customer tenure analysis
    """
    # Create tenure bins
    df_copy = df.copy()
    df_copy["Tenure Bins"] = pd.cut(
        df_copy["Customer Tenure in months"],
        bins=[0, 6, 12, 24, 36, 60, 100],
        labels=[
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-36 months",
            "37-60 months",
            "60+ months",
        ],
    )

    # Calculate churn rate by tenure
    tenure_analysis = (
        df_copy.groupby("Tenure Bins")
        .agg(
            {
                "Customer ID": "nunique",
                "Customer Churn Status": lambda x: (x == "Yes").sum(),
            }
        )
        .reset_index()
    )
    tenure_analysis["Churn Rate"] = (
        tenure_analysis["Customer Churn Status"] / tenure_analysis["Customer ID"]
    ) * 100

    # Create subplot
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Customer Distribution by Tenure", "Churn Rate by Tenure"),
        vertical_spacing=0.15,
    )

    # Customer distribution
    fig.add_trace(
        go.Bar(
            x=tenure_analysis["Tenure Bins"],
            y=tenure_analysis["Customer ID"],
            name="Total Customers",
            marker_color="lightblue",
        ),
        row=1,
        col=1,
    )

    # Churn rate
    fig.add_trace(
        go.Scatter(
            x=tenure_analysis["Tenure Bins"],
            y=tenure_analysis["Churn Rate"],
            mode="lines+markers",
            name="Churn Rate (%)",
            line=dict(color="red", width=3),
            marker=dict(size=8),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Customer Tenure Analysis", height=600, showlegend=False
    )

    fig.update_xaxes(title_text="Tenure Period", row=2, col=1)
    fig.update_yaxes(title_text="Number of Customers", row=1, col=1)
    fig.update_yaxes(title_text="Churn Rate (%)", row=2, col=1)

    return fig


def create_satisfaction_vs_churn(df):
    """
    Create satisfaction rate vs churn analysis
    """
    satisfaction_churn = (
        df.groupby(["Satisfaction Rate", "Customer Churn Status"])
        .size()
        .unstack(fill_value=0)
    )
    satisfaction_churn_pct = (
        satisfaction_churn.div(satisfaction_churn.sum(axis=1), axis=0) * 100
    )

    fig = px.bar(
        satisfaction_churn_pct,
        title="Churn Rate by Customer Satisfaction Level",
        labels={"value": "Percentage (%)", "index": "Satisfaction Rating"},
        color_discrete_map={"No": "#90EE90", "Yes": "#FF6B6B"},
    )

    fig.update_layout(
        xaxis_title="Satisfaction Rating (1-5)",
        yaxis_title="Percentage (%)",
        legend_title="Churn Status",
    )

    return fig


def create_revenue_analysis(df):
    """
    Create revenue analysis visualization
    """
    # Revenue by churn status
    revenue_analysis = (
        df.groupby("Customer Churn Status")["Total Revenue"]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Total Revenue by Churn Status",
            "Average Revenue per Customer",
            "Revenue Distribution (Retained)",
            "Revenue Distribution (Churned)",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "histogram"}],
        ],
    )

    # Total revenue
    fig.add_trace(
        go.Bar(
            x=revenue_analysis["Customer Churn Status"],
            y=revenue_analysis["sum"],
            name="Total Revenue",
            marker_color=["#90EE90", "#FF6B6B"],
        ),
        row=1,
        col=1,
    )

    # Average revenue
    fig.add_trace(
        go.Bar(
            x=revenue_analysis["Customer Churn Status"],
            y=revenue_analysis["mean"],
            name="Average Revenue",
            marker_color=["#90EE90", "#FF6B6B"],
        ),
        row=1,
        col=2,
    )

    # Revenue distribution for retained customers
    retained_revenue = df[df["Customer Churn Status"] == "No"]["Total Revenue"]
    fig.add_trace(
        go.Histogram(
            x=retained_revenue,
            name="Retained",
            marker_color="#90EE90",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2,
        col=1,
    )

    # Revenue distribution for churned customers
    churned_revenue = df[df["Customer Churn Status"] == "Yes"]["Total Revenue"]
    fig.add_trace(
        go.Histogram(
            x=churned_revenue,
            name="Churned",
            marker_color="#FF6B6B",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text="Revenue Analysis by Churn Status", height=800, showlegend=False
    )

    return fig


def create_geographic_analysis(df):
    """
    Create geographic analysis of churn
    """
    # State-wise churn analysis
    state_analysis = (
        df.groupby("State")
        .agg(
            {
                "Customer ID": "nunique",
                "Customer Churn Status": lambda x: (x == "Yes").sum(),
                "Total Revenue": "sum",
            }
        )
        .reset_index()
    )
    state_analysis["Churn Rate"] = (
        state_analysis["Customer Churn Status"] / state_analysis["Customer ID"]
    ) * 100
    state_analysis = state_analysis.sort_values("Churn Rate", ascending=False)

    # Top 15 states by churn rate
    top_states = state_analysis.head(15)

    fig = px.bar(
        top_states,
        x="State",
        y="Churn Rate",
        title="Top 15 States by Churn Rate",
        labels={"Churn Rate": "Churn Rate (%)"},
        color="Churn Rate",
        color_continuous_scale="Reds",
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500)

    return fig


def create_device_subscription_analysis(df):
    """
    Create device and subscription plan analysis
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Churn Rate by Device Type",
            "Top Subscription Plans by Churn Rate",
        ),
        vertical_spacing=0.15,
    )

    # Device analysis
    device_churn = (
        df.groupby(["MTN Device", "Customer Churn Status"]).size().unstack(fill_value=0)
    )
    device_churn_pct = device_churn.div(device_churn.sum(axis=1), axis=0) * 100

    fig.add_trace(
        go.Bar(
            x=device_churn_pct.index,
            y=device_churn_pct["Yes"],
            name="Churn Rate by Device",
            marker_color="orange",
        ),
        row=1,
        col=1,
    )

    # Subscription plan analysis (top 10)
    plan_churn = (
        df.groupby(["Subscription Plan", "Customer Churn Status"])
        .size()
        .unstack(fill_value=0)
    )
    plan_churn["Churn Rate"] = (
        plan_churn["Yes"] / (plan_churn["Yes"] + plan_churn["No"]) * 100
    )
    top_plans = plan_churn.sort_values("Churn Rate", ascending=False).head(10)

    fig.add_trace(
        go.Bar(
            x=top_plans.index,
            y=top_plans["Churn Rate"],
            name="Churn Rate by Plan",
            marker_color="purple",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Device and Subscription Analysis", height=700, showlegend=False
    )

    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_yaxes(title_text="Churn Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Churn Rate (%)", row=2, col=1)

    return fig


def create_correlation_heatmap(df):
    """
    Create correlation heatmap for numeric variables
    """
    # Select numeric columns
    numeric_cols = [
        "Age",
        "Satisfaction Rate",
        "Customer Tenure in months",
        "Unit Price",
        "Number of Times Purchased",
        "Total Revenue",
        "Data Usage",
    ]

    # Create binary churn variable
    df_corr = df[numeric_cols].copy()
    df_corr["Churn"] = (df["Customer Churn Status"] == "Yes").astype(int)

    # Calculate correlation matrix
    correlation_matrix = df_corr.corr()

    fig = px.imshow(
        correlation_matrix,
        title="Correlation Matrix of Key Variables",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=".2f",
    )

    fig.update_layout(height=600)

    return fig


def create_customer_lifecycle_analysis(df):
    """
    Create customer lifecycle analysis
    """
    # Create customer lifecycle bins based on tenure and revenue
    df_copy = df.copy()

    # Categorize customers
    df_copy["Revenue Category"] = pd.qcut(
        df_copy["Total Revenue"], q=3, labels=["Low", "Medium", "High"]
    )
    df_copy["Tenure Category"] = pd.cut(
        df_copy["Customer Tenure in months"],
        bins=[0, 12, 36, 100],
        labels=["New (0-12m)", "Established (1-3y)", "Loyal (3y+)"],
    )

    # Create lifecycle matrix
    lifecycle_matrix = (
        df_copy.groupby(
            ["Tenure Category", "Revenue Category", "Customer Churn Status"]
        )
        .size()
        .unstack(fill_value=0)
    )
    lifecycle_churn_rate = (
        lifecycle_matrix["Yes"]
        / (lifecycle_matrix["Yes"] + lifecycle_matrix["No"])
        * 100
    )
    lifecycle_churn_rate = lifecycle_churn_rate.reset_index()
    lifecycle_pivot = lifecycle_churn_rate.pivot(
        index="Tenure Category", columns="Revenue Category", values=0
    )

    fig = px.imshow(
        lifecycle_pivot,
        title="Customer Churn Rate by Lifecycle Stage",
        labels={
            "x": "Revenue Category",
            "y": "Tenure Category",
            "color": "Churn Rate (%)",
        },
        color_continuous_scale="Reds",
        text_auto=".1f",
    )

    fig.update_layout(height=400)

    return fig
