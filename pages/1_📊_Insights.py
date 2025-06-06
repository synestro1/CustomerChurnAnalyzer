import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import load_and_process_data, get_churn_insights

st.set_page_config(
    page_title="Customer Insights - MTN Churn Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Insights")
st.markdown("---")

try:
    # Load data
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        # Section des idées clés
        st.header("🎯 Key Findings")
        
        # Calculer les informations
        insights = get_churn_insights(df)
        
        # Afficher les indicateurs clés
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Churn Rate", f"{insights['churn_rate']:.1f}%")
        
        with col2:
            st.metric("Avg Satisfaction (Churned)", f"{insights['avg_satisfaction_churned']:.1f}")
        
        with col3:
            st.metric("Top Churn Reason", insights['top_churn_reason'])
        
        with col4:
            st.metric("High-Risk States", f"{len(insights['high_churn_states'])}")
        
        # Informations détaillées
        st.header("🔍 Detailed Analysis")
        
        # Taux de désabonnement par rapport au taux de satisfaction
        st.subheader("Customer Satisfaction vs Churn")
        satisfaction_churn = df.groupby(['Satisfaction Rate', 'Customer Churn Status']).size().unstack(fill_value=0)
        satisfaction_churn_pct = satisfaction_churn.div(satisfaction_churn.sum(axis=1), axis=0) * 100
        
        fig_satisfaction = px.bar(
            satisfaction_churn_pct,
            title="Churn Rate by Customer Satisfaction Level",
            labels={'value': 'Percentage (%)', 'index': 'Satisfaction Rating'},
            color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'}
        )
        st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        st.markdown("""
        **Key Insight:** Les clients dont le taux de satisfaction est faible (1-2) ont un taux de désabonnement nettement plus élevé.
        Cela indique que l'amélioration de la satisfaction des clients devrait être une priorité pour la fidélisation.
        """)
        
        # Analyse des motifs de désabonnement
        st.subheader("Primary Churn Reasons")
        churn_reasons = df[df['Customer Churn Status'] == 'Yes']['Reasons for Churn'].value_counts()
        
        fig_reasons = px.pie(
            values=churn_reasons.values,
            names=churn_reasons.index,
            title="Distribution of Churn Reasons"
        )
        st.plotly_chart(fig_reasons, use_container_width=True)
        
        # Analyse du taux de désabonnement par État
        st.subheader("Churn Analysis by State")
        state_analysis = df.groupby('State').agg({
            'Customer ID': 'nunique',
            'Customer Churn Status': lambda x: (x == 'Yes').sum()
        }).reset_index()
        state_analysis['Churn Rate'] = (state_analysis['Customer Churn Status'] / state_analysis['Customer ID']) * 100
        state_analysis = state_analysis.sort_values('Churn Rate', ascending=False)
        
        fig_states = px.bar(
            state_analysis.head(15),
            x='State',
            y='Churn Rate',
            title="Top 15 States by Churn Rate",
            labels={'Churn Rate': 'Churn Rate (%)'}
        )
        fig_states.update_xaxes(tickangle=45)
        st.plotly_chart(fig_states, use_container_width=True)
        
        # Device type analysis
        st.subheader("Churn by Device Type")
        device_churn = df.groupby(['MTN Device', 'Customer Churn Status']).size().unstack(fill_value=0)
        device_churn_pct = device_churn.div(device_churn.sum(axis=1), axis=0) * 100
        
        fig_device = px.bar(
            device_churn_pct,
            title="Churn Rate by Device Type",
            labels={'value': 'Percentage (%)', 'index': 'Device Type'},
            color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'}
        )
        st.plotly_chart(fig_device, use_container_width=True)
        
        # Advanced Business Intelligence from ML Analysis
        st.header("🧠 Advanced Business Intelligence")
        
        # Revenue per month analysis
        df['Revenue_per_Month'] = df.apply(
            lambda row: row['Total Revenue'] / row['Customer Tenure in months'] 
            if row['Customer Tenure in months'] > 0 else 0, axis=1
        )
        
        # Customer lifecycle analysis
        st.subheader("👥 Customer Lifecycle Analysis")
        
        # Create customer segments
        df['Revenue_Category'] = pd.qcut(df['Total Revenue'], q=3, labels=['Low', 'Medium', 'High'])
        df['Tenure_Category'] = pd.cut(df['Customer Tenure in months'], 
                                     bins=[0, 12, 36, 100], 
                                     labels=['New (0-12m)', 'Established (1-3y)', 'Loyal (3y+)'])
        
        # Lifecycle matrix
        lifecycle_analysis = df.groupby(['Tenure_Category', 'Revenue_Category', 'Customer Churn Status']).size().unstack(fill_value=0)
        if 'Yes' in lifecycle_analysis.columns and 'No' in lifecycle_analysis.columns:
            lifecycle_churn_rate = lifecycle_analysis['Yes'] / (lifecycle_analysis['Yes'] + lifecycle_analysis['No']) * 100
            lifecycle_pivot = lifecycle_churn_rate.reset_index().pivot(index='Tenure_Category', columns='Revenue_Category', values=0)
            
            fig_lifecycle = px.imshow(
                lifecycle_pivot,
                title="Churn Rate by Customer Lifecycle Stage (%)",
                labels={'x': 'Revenue Category', 'y': 'Tenure Category', 'color': 'Churn Rate (%)'},
                color_continuous_scale='Reds',
                text_auto='.1f'
            )
            st.plotly_chart(fig_lifecycle, use_container_width=True)
        
        # Revenue impact analysis
        st.subheader("💰 Revenue Impact Deep Dive")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Average revenue per month by churn status
            avg_rev_churned = df[df['Customer Churn Status'] == 'Yes']['Revenue_per_Month'].mean()
            avg_rev_retained = df[df['Customer Churn Status'] == 'No']['Revenue_per_Month'].mean()
            st.metric("Avg Monthly Revenue (Churned)", f"₦{avg_rev_churned:,.0f}")
        
        with col2:
            st.metric("Avg Monthly Revenue (Retained)", f"₦{avg_rev_retained:,.0f}")
        
        with col3:
            revenue_gap = avg_rev_retained - avg_rev_churned
            st.metric("Revenue Gap", f"₦{revenue_gap:,.0f}", f"{((revenue_gap/avg_rev_churned)*100):+.1f}%")
        
        # Data usage patterns
        st.subheader("📊 Data Usage Behavioral Patterns")
        
        # Data usage vs churn
        fig_data_usage = px.box(
            df, 
            x='Customer Churn Status', 
            y='Data Usage',
            title="Data Usage Distribution by Churn Status",
            labels={'Data Usage': 'Data Usage (GB)'}
        )
        st.plotly_chart(fig_data_usage, use_container_width=True)
        
        # Data usage efficiency (Revenue per GB)
        df['Revenue_per_GB'] = df['Total Revenue'] / (df['Data Usage'] + 0.01)  # Add small value to avoid division by zero
        
        usage_efficiency = df.groupby('Customer Churn Status')['Revenue_per_GB'].agg(['mean', 'median']).round(0)
        st.write("**Revenue Efficiency per GB of Data:**")
        st.dataframe(usage_efficiency, use_container_width=True)
        
        # Recommendations section with ML insights
        st.header("💡 Strategic Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Immediate Actions (Data-Driven)
            
            1. **Target New High-Revenue Customers**
               - Focus retention on customers with <12 months tenure and high revenue
               - These show highest churn risk in lifecycle analysis
            
            2. **Network Quality Investment**
               - Priority: States with >30% churn rate
               - Address "Poor Network" as top churn reason
            
            3. **Pricing Strategy Optimization**
               - Review plans with revenue/GB efficiency below median
               - Create competitive response to "Better Offers" churn reason
            
            4. **Data Plan Restructuring**
               - Address "Fast Data Consumption" complaints
               - Offer value-added services for high data users
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Long-term Strategy (ML-Enhanced)
            
            1. **Predictive Retention Model**
               - Deploy machine learning models for early churn detection
               - Target customers 3-6 months before predicted churn
            
            2. **Personalized Engagement**
               - Use revenue per month patterns for customized offers
               - Lifecycle-based communication strategies
            
            3. **Service Quality Improvement**
               - Continuous network quality monitoring
               - Proactive customer service for satisfaction <3
            
            4. **Customer Lifetime Value Optimization**
               - Focus on converting New customers to Established
               - Reward Loyal customers to prevent competitive switching
            """)
        
        # Risk assessment
        st.header("⚠️ Risk Assessment")
        
        # Calculate revenue at risk
        churned_revenue = df[df['Customer Churn Status'] == 'Yes']['Total Revenue'].sum()
        total_revenue = df['Total Revenue'].sum()
        revenue_at_risk_pct = (churned_revenue / total_revenue) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Revenue Lost to Churn", f"₦{churned_revenue:,.0f}")
        
        with col2:
            st.metric("% of Total Revenue", f"{revenue_at_risk_pct:.1f}%")
        
        with col3:
            potential_savings = churned_revenue * 0.3  # Assume 30% retention possible
            st.metric("Potential Savings", f"₦{potential_savings:,.0f}")
        
    else:
        st.error("No data available for analysis.")
        
except Exception as e:
    st.error(f"Error loading insights: {str(e)}")
    st.info("Please check the data source and try again.")
