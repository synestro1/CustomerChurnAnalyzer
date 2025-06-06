import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from utils.data_processor import load_and_process_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ML Models - MTN Churn Analysis",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Models")
st.markdown("---")

# Custom CSS for better styling
st.markdown("""
<style>
    .model-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFCC00;
        margin: 0.5rem 0;
    }
    .metric-highlight {
        background-color: #FFF9E6;
        padding: 0.5rem;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

try:
    # Load data
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        st.header("🔧 Model Training & Evaluation")
        
        # Data preparation section
        with st.expander("📊 Data Preparation Details", expanded=False):
            st.subheader("Feature Engineering")
            
            # Create Revenue per Month feature
            df['Revenue_per_Month'] = df.apply(
                lambda row: row['Total Revenue'] / row['Customer Tenure in months'] 
                if row['Customer Tenure in months'] > 0 else 0, axis=1
            )
            
            st.write("**Created Features:**")
            st.write("- Revenue per Month: Total Revenue / Customer Tenure")
            
            # Show feature statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Features Available", len(df.columns))
            with col2:
                st.metric("Revenue/Month (Avg)", f"₦{df['Revenue_per_Month'].mean():,.0f}")
        
        # Model training section
        st.header("🎯 Model Training Results")
        
        # Prepare data for modeling
        @st.cache_data
        def prepare_model_data(dataframe):
            # Define target column
            target_col = 'Customer Churn Status'
            
            # Columns to drop for modeling
            cols_to_drop = [target_col, 'Date of Purchase', 'Customer ID', 'Full Name', 
                           'Customer Review', 'Reasons for Churn']
            existing_drop_cols = [col for col in cols_to_drop if col in dataframe.columns]
            predictor_cols = [col for col in dataframe.columns if col not in existing_drop_cols]
            
            # Create feature matrix
            df_features = dataframe[predictor_cols].copy()
            
            # One-hot encoding for categorical variables
            df_encoded = pd.get_dummies(df_features, drop_first=True)
            X = df_encoded.copy()
            
            # Create target variable
            y = dataframe[target_col].apply(lambda x: 1 if isinstance(x, str) and x.strip().lower() == 'yes' else 0)
            
            return X, y
        
        @st.cache_data
        def train_models(X, y):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
            
            results = {}
            
            # Train and evaluate models
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'confusion_matrix': cm,
                    'classification_report': report,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
            
            return results, scaler, X_train.columns
        
        # Prepare and train models
        X, y = prepare_model_data(df)
        
        st.info(f"Training models on {len(X)} samples with {len(X.columns)} features...")
        
        # Train models
        model_results, trained_scaler, feature_names = train_models(X, y)
        
        # Display model comparison
        st.subheader("📈 Model Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for name, results in model_results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision (Churn)': results['classification_report']['1']['precision'],
                'Recall (Churn)': results['classification_report']['1']['recall'],
                'F1-Score (Churn)': results['classification_report']['1']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Display comparison table
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
        # Best model highlight
        best_model = comparison_df.iloc[0]
        st.success(f"🏆 Best Performing Model: **{best_model['Model']}** with {best_model['Accuracy']:.1%} accuracy")
        
        # Detailed model analysis
        st.subheader("🔍 Detailed Model Analysis")
        
        # Model selection for detailed view
        selected_model = st.selectbox(
            "Select a model for detailed analysis:",
            options=list(model_results.keys()),
            index=0
        )
        
        if selected_model:
            model_data = model_results[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                cm = model_data['confusion_matrix']
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f"Confusion Matrix - {selected_model}",
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # Classification metrics
                report = model_data['classification_report']
                
                st.markdown(f"""
                <div class="model-card">
                    <h4>{selected_model} - Performance Metrics</h4>
                    <div class="metric-highlight">
                        Accuracy: {model_data['accuracy']:.1%}
                    </div>
                    <br>
                    <strong>No Churn (Class 0):</strong><br>
                    • Precision: {report['0']['precision']:.3f}<br>
                    • Recall: {report['0']['recall']:.3f}<br>
                    • F1-Score: {report['0']['f1-score']:.3f}<br>
                    <br>
                    <strong>Churn (Class 1):</strong><br>
                    • Precision: {report['1']['precision']:.3f}<br>
                    • Recall: {report['1']['recall']:.3f}<br>
                    • F1-Score: {report['1']['f1-score']:.3f}<br>
                    <br>
                    <strong>Overall:</strong><br>
                    • Macro Avg F1: {report['macro avg']['f1-score']:.3f}<br>
                    • Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}
                </div>
                """, unsafe_allow_html=True)
        
        # Feature importance (for tree-based models)
        if selected_model in ['Random Forest', 'Decision Tree']:
            st.subheader("🌟 Feature Importance")
            
            model = model_results[selected_model]['model']
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 15 Feature Importance - {selected_model}",
                    labels={'Importance': 'Feature Importance Score'}
                )
                fig_importance.update_layout(height=500)
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model insights and recommendations
        st.header("💡 ML Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Key ML Findings
            
            **Model Performance:**
            - Most models achieve 70-75% accuracy
            - Random Forest and Logistic Regression show consistent performance
            - Tree-based models provide interpretable feature importance
            
            **Class Imbalance Impact:**
            - Churn class (minority) is harder to predict
            - Precision-Recall trade-off is important
            - Consider cost-sensitive learning for business impact
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Business Applications
            
            **Predictive Targeting:**
            - Deploy models for real-time churn prediction
            - Focus retention efforts on high-risk customers
            - Automate early warning systems
            
            **Strategic Insights:**
            - Revenue per month is a strong predictor
            - Customer tenure strongly influences retention
            - Device type and subscription plans matter
            """)
        
        # Advanced analytics section
        st.header("📊 Advanced Analytics")
        
        # ROC Curve analysis
        st.subheader("ROC Curve Analysis")
        
        fig_roc = go.Figure()
        
        for name, results in model_results.items():
            model = results['model']
            y_test = results['y_test']
            
            if hasattr(model, 'predict_proba'):
                # Get probabilities for the positive class
                y_prob = model.predict_proba(trained_scaler.transform(X.iloc[y_test.index]))[:, 1]
                
                # Calculate ROC curve points
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{name} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title='ROC Curves - Model Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Business impact simulation
        st.subheader("💰 Business Impact Simulation")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate potential savings
        total_customers = len(df)
        churned_customers = len(df[df['Customer Churn Status'] == 'Yes'])
        avg_customer_value = df['Total Revenue'].mean()
        
        # Simulate intervention scenarios
        best_model_accuracy = max([results['accuracy'] for results in model_results.values()])
        
        with col1:
            st.metric(
                "Customers at Risk",
                f"{churned_customers:,}",
                f"{(churned_customers/total_customers)*100:.1f}% of total"
            )
        
        with col2:
            potential_savings = churned_customers * avg_customer_value * 0.3  # Assume 30% retention with intervention
            st.metric(
                "Potential Revenue Recovery",
                f"₦{potential_savings:,.0f}",
                "30% intervention success"
            )
        
        with col3:
            ml_enhanced_savings = potential_savings * best_model_accuracy
            st.metric(
                "ML-Enhanced Recovery",
                f"₦{ml_enhanced_savings:,.0f}",
                f"With {best_model_accuracy:.1%} targeting accuracy"
            )
        
        # Export model results
        st.header("📤 Model Export & Deployment")
        
        st.info("""
        **Model Deployment Recommendations:**
        
        1. **Production Pipeline:** Implement the Random Forest or Logistic Regression model
        2. **Real-time Scoring:** Set up API endpoints for live customer scoring
        3. **Batch Processing:** Schedule regular model updates and customer risk scoring
        4. **Monitoring:** Track model performance and data drift over time
        5. **A/B Testing:** Test intervention strategies on predicted high-risk customers
        """)
        
    else:
        st.error("No data available for machine learning analysis.")

except Exception as e:
    st.error(f"Error in machine learning analysis: {str(e)}")
    st.info("Please check the data source and model dependencies.")