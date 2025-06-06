import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import load_and_process_data

st.set_page_config(
    page_title="Data Dashboard - MTN Churn Analysis",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Data Dashboard")
st.markdown("---")

try:
    # Load data
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        # Data overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Unique Customers", f"{df['Customer ID'].nunique():,}")
        
        with col3:
            st.metric("Data Points", f"{df.shape[1]}")
        
        with col4:
            missing_data = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_data:,}")
        
        # Advanced filters
        st.header("🔧 Advanced Data Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Customer status filter
            churn_status = st.selectbox(
                "Churn Status",
                options=['All', 'Yes', 'No'],
                index=0
            )
            
            # Age range filter
            age_range = st.slider(
                "Age Range",
                min_value=int(df['Age'].min()),
                max_value=int(df['Age'].max()),
                value=(int(df['Age'].min()), int(df['Age'].max()))
            )
        
        with col2:
            # State filter
            selected_states = st.multiselect(
                "Select States",
                options=sorted(df['State'].unique()),
                default=[]
            )
            
            # Device type filter
            selected_devices = st.multiselect(
                "Select Device Types",
                options=df['MTN Device'].unique(),
                default=[]
            )
        
        with col3:
            # Revenue range filter
            revenue_range = st.slider(
                "Total Revenue Range (₦)",
                min_value=float(df['Total Revenue'].min()),
                max_value=float(df['Total Revenue'].max()),
                value=(float(df['Total Revenue'].min()), float(df['Total Revenue'].max())),
                format="%.0f"
            )
            
            # Tenure filter
            tenure_range = st.slider(
                "Customer Tenure (months)",
                min_value=int(df['Customer Tenure in months'].min()),
                max_value=int(df['Customer Tenure in months'].max()),
                value=(int(df['Customer Tenure in months'].min()), int(df['Customer Tenure in months'].max()))
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if churn_status != 'All':
            filtered_df = filtered_df[filtered_df['Customer Churn Status'] == churn_status]
        
        filtered_df = filtered_df[
            (filtered_df['Age'] >= age_range[0]) & 
            (filtered_df['Age'] <= age_range[1])
        ]
        
        if selected_states:
            filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
        
        if selected_devices:
            filtered_df = filtered_df[filtered_df['MTN Device'].isin(selected_devices)]
        
        filtered_df = filtered_df[
            (filtered_df['Total Revenue'] >= revenue_range[0]) & 
            (filtered_df['Total Revenue'] <= revenue_range[1])
        ]
        
        filtered_df = filtered_df[
            (filtered_df['Customer Tenure in months'] >= tenure_range[0]) & 
            (filtered_df['Customer Tenure in months'] <= tenure_range[1])
        ]
        
        # Display filtered results
        st.header(f"📈 Filtered Results ({len(filtered_df):,} records)")
        
        if not filtered_df.empty:
            # Summary statistics for filtered data
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                avg_age = filtered_df['Age'].mean()
                st.metric("Avg Age", f"{avg_age:.1f}")
            
            with col2:
                avg_satisfaction = filtered_df['Satisfaction Rate'].mean()
                st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}")
            
            with col3:
                avg_tenure = filtered_df['Customer Tenure in months'].mean()
                st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
            
            with col4:
                total_revenue = filtered_df['Total Revenue'].sum()
                st.metric("Total Revenue", f"₦{total_revenue:,.0f}")
            
            with col5:
                if len(filtered_df) > 0:
                    churn_rate = (filtered_df['Customer Churn Status'] == 'Yes').mean() * 100
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                else:
                    st.metric("Churn Rate", "N/A")
            
            # Data exploration tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Stats", "📋 Raw Data", "📈 Quick Charts", "🔍 Customer Search"])
            
            with tab1:
                st.subheader("Statistical Summary")
                
                # Numeric columns summary
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                st.write("**Numeric Variables Summary:**")
                st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)
                
                # Categorical columns summary
                st.write("**Categorical Variables Summary:**")
                categorical_cols = ['State', 'MTN Device', 'Gender', 'Customer Review', 'Subscription Plan', 'Customer Churn Status']
                
                for col in categorical_cols:
                    if col in filtered_df.columns:
                        st.write(f"**{col}:**")
                        value_counts = filtered_df[col].value_counts()
                        st.write(value_counts.head(10))
                        st.write("---")
            
            with tab2:
                st.subheader("Raw Data View")
                
                # Column selector
                all_columns = filtered_df.columns.tolist()
                selected_columns = st.multiselect(
                    "Select columns to display:",
                    options=all_columns,
                    default=all_columns[:8]  # Show first 8 columns by default
                )
                
                if selected_columns:
                    # Pagination
                    rows_per_page = st.selectbox("Rows per page:", [10, 25, 50, 100], index=1)
                    total_rows = len(filtered_df)
                    total_pages = (total_rows - 1) // rows_per_page + 1
                    
                    if total_pages > 1:
                        page = st.selectbox("Page:", range(1, total_pages + 1))
                        start_idx = (page - 1) * rows_per_page
                        end_idx = min(start_idx + rows_per_page, total_rows)
                    else:
                        start_idx = 0
                        end_idx = total_rows
                    
                    st.dataframe(
                        filtered_df[selected_columns].iloc[start_idx:end_idx],
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = filtered_df[selected_columns].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Data as CSV",
                        data=csv,
                        file_name="mtn_churn_filtered_data.csv",
                        mime="text/csv"
                    )
            
            with tab3:
                st.subheader("Quick Data Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Variable selection for charts
                    chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Histogram", "Box Plot"])
                    
                    if chart_type == "Bar Chart":
                        categorical_columns = filtered_df.select_dtypes(include=['object']).columns
                        selected_cat = st.selectbox("Select Categorical Variable:", categorical_columns)
                        
                        if selected_cat:
                            value_counts = filtered_df[selected_cat].value_counts().head(10)
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Distribution of {selected_cat}",
                                labels={'x': selected_cat, 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Histogram":
                        numeric_columns = filtered_df.select_dtypes(include=['int64', 'float64']).columns
                        selected_num = st.selectbox("Select Numeric Variable:", numeric_columns)
                        
                        if selected_num:
                            fig = px.histogram(
                                filtered_df,
                                x=selected_num,
                                title=f"Distribution of {selected_num}",
                                nbins=30
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Box Plot":
                        numeric_columns = filtered_df.select_dtypes(include=['int64', 'float64']).columns
                        selected_num = st.selectbox("Select Numeric Variable:", numeric_columns)
                        categorical_columns = filtered_df.select_dtypes(include=['object']).columns
                        selected_cat = st.selectbox("Group by:", categorical_columns)
                        
                        if selected_num and selected_cat:
                            fig = px.box(
                                filtered_df,
                                x=selected_cat,
                                y=selected_num,
                                title=f"{selected_num} by {selected_cat}"
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Correlation heatmap for numeric variables
                    st.write("**Correlation Matrix:**")
                    numeric_df = filtered_df.select_dtypes(include=['int64', 'float64'])
                    if len(numeric_df.columns) > 1:
                        corr_matrix = numeric_df.corr()
                        fig_corr = px.imshow(
                            corr_matrix,
                            title="Correlation Heatmap",
                            color_continuous_scale='RdBu_r',
                            aspect='auto'
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
            
            with tab4:
                st.subheader("Customer Search & Analysis")
                
                # Customer search
                customer_id = st.text_input("Enter Customer ID:")
                
                if customer_id:
                    customer_data = filtered_df[filtered_df['Customer ID'] == customer_id]
                    
                    if not customer_data.empty:
                        st.success(f"Found {len(customer_data)} record(s) for Customer ID: {customer_id}")
                        
                        # Display customer details
                        for idx, row in customer_data.iterrows():
                            st.write("**Customer Details:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**Name:** {row['Full Name']}")
                                st.write(f"**Age:** {row['Age']}")
                                st.write(f"**Gender:** {row['Gender']}")
                                st.write(f"**State:** {row['State']}")
                            
                            with col2:
                                st.write(f"**Device:** {row['MTN Device']}")
                                st.write(f"**Plan:** {row['Subscription Plan']}")
                                st.write(f"**Tenure:** {row['Customer Tenure in months']} months")
                                st.write(f"**Satisfaction:** {row['Satisfaction Rate']}")
                            
                            with col3:
                                st.write(f"**Total Revenue:** ₦{row['Total Revenue']:,.0f}")
                                st.write(f"**Data Usage:** {row['Data Usage']:.2f} GB")
                                st.write(f"**Churn Status:** {row['Customer Churn Status']}")
                                if row['Customer Churn Status'] == 'Yes':
                                    st.write(f"**Churn Reason:** {row['Reasons for Churn']}")
                            
                            st.write("---")
                    else:
                        st.error("Customer ID not found in the filtered data.")
        
        else:
            st.warning("No data matches the selected filters. Please adjust your filter criteria.")
    
    else:
        st.error("No data available for the dashboard.")

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.info("Please check the data source and try again.")
