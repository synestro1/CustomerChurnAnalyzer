import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_processor import load_and_process_data
from utils.visualization_helpers import create_churn_distribution, create_age_analysis, create_tenure_analysis

st.set_page_config(
    page_title="Visualizations - MTN Churn Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Interactive Visualizations")
st.markdown("---")

try:
    # Load data
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        # Sidebar filters
        st.sidebar.header("🔧 Filters")
        
        # State filter
        states = st.sidebar.multiselect(
            "Select States",
            options=df['State'].unique(),
            default=df['State'].unique()[:5]  # Default to first 5 states
        )
        
        # Device filter
        devices = st.sidebar.multiselect(
            "Select Device Types",
            options=df['MTN Device'].unique(),
            default=df['MTN Device'].unique()
        )
        
        # Satisfaction filter
        satisfaction_range = st.sidebar.slider(
            "Customer Satisfaction Range",
            min_value=int(df['Satisfaction Rate'].min()),
            max_value=int(df['Satisfaction Rate'].max()),
            value=(int(df['Satisfaction Rate'].min()), int(df['Satisfaction Rate'].max()))
        )
        
        # Filter data based on selections
        filtered_df = df[
            (df['State'].isin(states)) &
            (df['MTN Device'].isin(devices)) &
            (df['Satisfaction Rate'] >= satisfaction_range[0]) &
            (df['Satisfaction Rate'] <= satisfaction_range[1])
        ]
        
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Please adjust your selections.")
        else:
            # Overview metrics for filtered data
            st.header("📊 Filtered Data Overview")
            
            total_customers = filtered_df['Customer ID'].nunique()
            churned_customers = filtered_df[filtered_df['Customer Churn Status'] == 'Yes']['Customer ID'].nunique()
            churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Customers in Selection", f"{total_customers:,}")
            with col2:
                st.metric("Churned Customers", f"{churned_customers:,}")
            with col3:
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            # Churn Distribution
            st.header("🎯 Churn Distribution Analysis")
            fig_churn_dist = create_churn_distribution(filtered_df)
            st.plotly_chart(fig_churn_dist, use_container_width=True)
            
            # Age and Gender Analysis
            st.header("👥 Demographics Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution by churn
                fig_age = create_age_analysis(filtered_df)
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                # Gender analysis
                gender_churn = filtered_df.groupby(['Gender', 'Customer Churn Status']).size().unstack(fill_value=0)
                gender_churn_pct = gender_churn.div(gender_churn.sum(axis=1), axis=0) * 100
                
                fig_gender = px.bar(
                    gender_churn_pct,
                    title="Churn Rate by Gender",
                    labels={'value': 'Percentage (%)', 'index': 'Gender'},
                    color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'}
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            
            # Customer Tenure Analysis
            st.header("⏱️ Customer Tenure Analysis")
            fig_tenure = create_tenure_analysis(filtered_df)
            st.plotly_chart(fig_tenure, use_container_width=True)
            
            # Revenue Analysis
            st.header("💰 Revenue Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue by churn status
                revenue_by_churn = filtered_df.groupby('Customer Churn Status')['Total Revenue'].sum()
                fig_revenue = px.pie(
                    values=revenue_by_churn.values,
                    names=revenue_by_churn.index,
                    title="Revenue Distribution by Churn Status",
                    color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'}
                )
                st.plotly_chart(fig_revenue, use_container_width=True)
            
            with col2:
                # Average revenue by device type
                avg_revenue_device = filtered_df.groupby('MTN Device')['Total Revenue'].mean().sort_values(ascending=False)
                fig_device_revenue = px.bar(
                    x=avg_revenue_device.index,
                    y=avg_revenue_device.values,
                    title="Average Revenue by Device Type",
                    labels={'x': 'Device Type', 'y': 'Average Revenue (₦)'}
                )
                fig_device_revenue.update_xaxes(tickangle=45)
                st.plotly_chart(fig_device_revenue, use_container_width=True)
            
            # Data Usage Analysis
            st.header("📊 Data Usage Patterns")
            
            # Create bins for data usage
            filtered_df['Data Usage Range'] = pd.cut(
                filtered_df['Data Usage'], 
                bins=5, 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            
            usage_churn = filtered_df.groupby(['Data Usage Range', 'Customer Churn Status']).size().unstack(fill_value=0)
            usage_churn_pct = usage_churn.div(usage_churn.sum(axis=1), axis=0) * 100
            
            fig_usage = px.bar(
                usage_churn_pct,
                title="Churn Rate by Data Usage Range",
                labels={'value': 'Percentage (%)', 'index': 'Data Usage Range'},
                color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'}
            )
            st.plotly_chart(fig_usage, use_container_width=True)
            
            # Subscription Plan Analysis
            st.header("📱 Subscription Plan Analysis")
            
            # Group similar plans for better visualization
            plan_churn = filtered_df.groupby(['Subscription Plan', 'Customer Churn Status']).size().unstack(fill_value=0)
            plan_churn['Churn Rate'] = plan_churn['Yes'] / (plan_churn['Yes'] + plan_churn['No']) * 100
            plan_churn = plan_churn.sort_values('Churn Rate', ascending=False).head(15)
            
            fig_plans = px.bar(
                x=plan_churn.index,
                y=plan_churn['Churn Rate'],
                title="Top 15 Subscription Plans by Churn Rate",
                labels={'x': 'Subscription Plan', 'y': 'Churn Rate (%)'}
            )
            fig_plans.update_xaxes(tickangle=45)
            st.plotly_chart(fig_plans, use_container_width=True)
            
            # Correlation Analysis
            st.header("🔗 Correlation Analysis")
            
            # Select numeric columns for correlation
            numeric_cols = ['Age', 'Satisfaction Rate', 'Customer Tenure in months', 
                          'Unit Price', 'Number of Times Purchased', 'Total Revenue', 'Data Usage']
            
            # Create binary churn variable for correlation
            corr_df = filtered_df[numeric_cols].copy()
            corr_df['Churn'] = (filtered_df['Customer Churn Status'] == 'Yes').astype(int)
            
            correlation_matrix = corr_df.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Correlation Matrix of Key Variables",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Advanced visualizations from ML analysis
            st.header("🔬 Advanced Analytics Visualizations")
            
            # Revenue per month analysis
            filtered_df['Revenue_per_Month'] = filtered_df.apply(
                lambda row: row['Total Revenue'] / row['Customer Tenure in months'] 
                if row['Customer Tenure in months'] > 0 else 0, axis=1
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue per month by churn status
                fig_rev_month = px.box(
                    filtered_df,
                    x='Customer Churn Status',
                    y='Revenue_per_Month',
                    title="Monthly Revenue Distribution by Churn Status",
                    labels={'Revenue_per_Month': 'Revenue per Month (₦)'},
                    color='Customer Churn Status',
                    color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'}
                )
                st.plotly_chart(fig_rev_month, use_container_width=True)
            
            with col2:
                # Customer lifecycle matrix
                filtered_df['Revenue_Category'] = pd.qcut(filtered_df['Total Revenue'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                filtered_df['Tenure_Category'] = pd.cut(filtered_df['Customer Tenure in months'], 
                                                       bins=[0, 12, 36, 100], 
                                                       labels=['New (0-12m)', 'Established (1-3y)', 'Loyal (3y+)'])
                
                # Calculate churn rates by lifecycle stage
                lifecycle_stats = filtered_df.groupby(['Tenure_Category', 'Revenue_Category']).agg({
                    'Customer Churn Status': lambda x: (x == 'Yes').sum(),
                    'Customer ID': 'count'
                }).reset_index()
                lifecycle_stats['Churn_Rate'] = (lifecycle_stats['Customer Churn Status'] / lifecycle_stats['Customer ID']) * 100
                
                if not lifecycle_stats.empty:
                    fig_lifecycle_bar = px.bar(
                        lifecycle_stats,
                        x='Tenure_Category',
                        y='Churn_Rate',
                        color='Revenue_Category',
                        title="Churn Rate by Customer Lifecycle Stage",
                        labels={'Churn_Rate': 'Churn Rate (%)', 'Tenure_Category': 'Tenure Category'},
                        barmode='group'
                    )
                    st.plotly_chart(fig_lifecycle_bar, use_container_width=True)
            
            # Revenue efficiency analysis
            st.subheader("💰 Revenue Efficiency Analysis")
            
            # Revenue per GB analysis
            filtered_df['Revenue_per_GB'] = filtered_df['Total Revenue'] / (filtered_df['Data Usage'] + 0.01)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue efficiency by device type
                efficiency_by_device = filtered_df.groupby('MTN Device')['Revenue_per_GB'].mean().sort_values(ascending=False)
                fig_efficiency = px.bar(
                    x=efficiency_by_device.index,
                    y=efficiency_by_device.values,
                    title="Revenue Efficiency by Device Type (₦/GB)",
                    labels={'x': 'Device Type', 'y': 'Revenue per GB (₦)'}
                )
                fig_efficiency.update_xaxes(tickangle=45)
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            with col2:
                # Satisfaction vs Revenue correlation
                fig_sat_rev = px.scatter(
                    filtered_df,
                    x='Satisfaction Rate',
                    y='Revenue_per_Month',
                    color='Customer Churn Status',
                    title="Customer Satisfaction vs Monthly Revenue",
                    labels={'Revenue_per_Month': 'Revenue per Month (₦)'},
                    color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'},
                    opacity=0.6
                )
                st.plotly_chart(fig_sat_rev, use_container_width=True)
            
            # Churn reason analysis (enhanced)
            st.subheader("🔍 Enhanced Churn Reason Analysis")
            
            # Top churn reasons with impact analysis
            churn_data = filtered_df[filtered_df['Customer Churn Status'] == 'Yes']
            if not churn_data.empty and 'Reasons for Churn' in churn_data.columns:
                reason_analysis = churn_data.groupby('Reasons for Churn').agg({
                    'Customer ID': 'count',
                    'Total Revenue': 'sum',
                    'Revenue_per_Month': 'mean'
                }).reset_index()
                reason_analysis.columns = ['Churn_Reason', 'Customer_Count', 'Total_Revenue_Lost', 'Avg_Monthly_Revenue']
                reason_analysis = reason_analysis.sort_values('Total_Revenue_Lost', ascending=False).head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Customer count by reason
                    fig_reason_count = px.bar(
                        reason_analysis,
                        x='Customer_Count',
                        y='Churn_Reason',
                        title="Top Churn Reasons by Customer Count",
                        orientation='h',
                        labels={'Customer_Count': 'Number of Customers'}
                    )
                    st.plotly_chart(fig_reason_count, use_container_width=True)
                
                with col2:
                    # Revenue impact by reason
                    fig_reason_revenue = px.bar(
                        reason_analysis,
                        x='Total_Revenue_Lost',
                        y='Churn_Reason',
                        title="Revenue Impact by Churn Reason",
                        orientation='h',
                        labels={'Total_Revenue_Lost': 'Revenue Lost (₦)'},
                        color='Total_Revenue_Lost',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_reason_revenue, use_container_width=True)
            
            # Predictive indicators visualization
            st.subheader("📈 Predictive Indicators")
            
            # Create risk score visualization
            # Calculate risk factors
            filtered_df['Risk_Score'] = 0
            filtered_df.loc[filtered_df['Satisfaction Rate'] <= 2, 'Risk_Score'] += 3
            filtered_df.loc[filtered_df['Customer Tenure in months'] <= 12, 'Risk_Score'] += 2
            filtered_df.loc[filtered_df['Revenue_per_Month'] < filtered_df['Revenue_per_Month'].median(), 'Risk_Score'] += 1
            
            # Risk score distribution
            fig_risk = px.histogram(
                filtered_df,
                x='Risk_Score',
                color='Customer Churn Status',
                title="Risk Score Distribution (Higher = More Risk)",
                labels={'Risk_Score': 'Risk Score', 'count': 'Number of Customers'},
                color_discrete_map={'No': '#90EE90', 'Yes': '#FF6B6B'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Summary insights based on visualizations
            st.header("🎯 Visualization Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **Principaux modèles observés :**
                - La satisfaction des clients montre une forte corrélation inverse avec le désabonnement
                - Le revenu par mois est un fort prédicteur de la fidélisation
                - Les nouveaux clients à revenu élevé sont les plus exposés au risque
                - Le type d'appareil affecte considérablement l'efficacité des revenus
                - Le regroupement géographique des désabonnements suggère des problèmes de qualité du réseau
                """)
            
            with col2:
                st.success("""
                **Insights exploitables :**
                - Concentrer la fidélisation sur les clients à haut risque (score ≥ 4)
                - Cibler les nouveaux clients avec un onboarding personnalisé
                - Optimiser les plans de données pour une meilleure efficacité des revenus
                - Aborder la qualité du réseau dans les états à fort taux de désabonnement
                - Mettre en œuvre un suivi de la satisfaction pour une intervention précoce
                """)
    
    else:
        st.error("No data available for visualization.")

except Exception as e:
    st.error(f"Error creating visualizations: {str(e)}")
    st.info("Please check the data source and try again.")
