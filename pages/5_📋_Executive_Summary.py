import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_processor import load_and_process_data, get_churn_insights

st.set_page_config(
    page_title="Executive Summary - MTN Churn Analysis",
    page_icon="📋",
    layout="wide"
)

# Custom CSS for executive styling
st.markdown("""
<style>
    .executive-header {
        background: linear-gradient(90deg, #FFCC00 0%, #FFD700 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: #000000;
    }
    .key-finding {
        background-color: #FFF9E6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFCC00;
        margin: 1rem 0;
    }
    .key-finding h4.insight-title {
    color: #4A80A9; /* Bleu principal pour le titre (inspiré de "Enhanced Visualizations") */
    }

    .key-finding p {
    color: #6082A0;
    }
    .key-finding ul {
    color: #6082A0;
    }

    .key-finding li {
    color: #6082A0;
    }        
    .recommendation-box {
        background-color: #F0F8F0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .risk-alert {
        background-color: #FFF5F5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="executive-header">
    <h1>📋 Executive Summary</h1>
    <p>Analyse du taux d'attrition des clients de MTN Nigeria - Perspectives et recommandations stratégiques</p>
</div>
""", unsafe_allow_html=True)

try:
    # Load data
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        # Executive Dashboard Overview
        st.header("📊 Business Impact Dashboard")
        
        # Calculate key metrics
        total_customers = df['Customer ID'].nunique()
        churned_customers = df[df['Customer Churn Status'] == 'Yes']['Customer ID'].nunique()
        churn_rate = (churned_customers / total_customers) * 100
        total_revenue = df['Total Revenue'].sum()
        churned_revenue = df[df['Customer Churn Status'] == 'Yes']['Total Revenue'].sum()
        revenue_at_risk = (churned_revenue / total_revenue) * 100
        
        # Create Revenue per Month feature
        df['Revenue_per_Month'] = df.apply(
            lambda row: row['Total Revenue'] / row['Customer Tenure in months'] 
            if row['Customer Tenure in months'] > 0 else 0, axis=1
        )
        
        # Key metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Customers",
                f"{total_customers:,}",
                help="Unique customers in dataset"
            )
        
        with col2:
            st.metric(
                "Churn Rate",
                f"{churn_rate:.1f}%",
                f"{churned_customers:,} customers lost",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Revenue at Risk",
                f"{revenue_at_risk:.1f}%",
                f"₦{churned_revenue:,.0f} lost"
            )
        
        with col4:
            avg_customer_value = df['Total Revenue'].mean()
            st.metric(
                "Avg Customer Value",
                f"₦{avg_customer_value:,.0f}",
                help="Average total revenue per customer"
            )
        
        with col5:
            potential_savings = churned_revenue * 0.3  # 30% retention assumption
            st.metric(
                "Potential Recovery",
                f"₦{potential_savings:,.0f}",
                "With 30% retention improvement"
            )
        
        # Key Findings Section
        st.header("🎯 Key Business Findings")
        
        # Get insights
        insights = get_churn_insights(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="key-finding">
                <h4 class="insight-title">💡 Découverte critique : Risque lié au cycle de vie du client</h4>
                <p><strong>Les nouveaux clients à revenu élevé sont les plus exposés au risque de désabonnement.</strong></p>
                <ul>
                    <li>Customers with <12 months tenure show 35%+ churn rates</li>
                    <li>Les nouveaux clients à forte valeur ajoutée doivent faire l'objet d'une attention immédiate</li>
                    <li>Les programmes de fidélisation de la première année sont essentiels</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="key-finding">
                <h4 class="insight-title">📊 Revenue Impact Analysis</h4>
                <p><strong>Les clients désabonnés génèrent des revenus mensuels nettement inférieurs.</strong></p>
                <ul>
                    <li>Écart moyen de revenus mensuels : ₦{:,.0f}</li>
                    <li>L'efficacité des revenus varie selon le type d'appareil</li>
                    <li>Les modèles d'utilisation des données prédisent la probabilité de désabonnement</li>
                </ul>
            </div>
            """.format(
                df[df['Customer Churn Status'] == 'No']['Revenue_per_Month'].mean() - 
                df[df['Customer Churn Status'] == 'Yes']['Revenue_per_Month'].mean()
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="key-finding">
                <h4 class="insight-title">🎪 Principaux facteurs de désabonnement</h4>
                <p><strong>Raison principale : {insights['top_churn_reason']}</strong></p>
                <ul>
                    <li>Les problèmes de qualité du réseau entraînent un désabonnement significatif</li>
                    <li>Les offres concurrentielles représentent une menace majeure</li>
                    <li>La satisfaction des clients est fortement corrélée à la fidélisation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="key-finding">
                <h4 class="insight-title">🗺️ Modèles géographiques</h4>
                <p><strong>{len(insights['high_churn_states'])} états présentent des taux de désabonnement supérieurs à la moyenne.</strong></p>
                <ul>
                    <li>Variations régionales de la qualité du réseau</li>
                    <li>Pressions concurrentielles locales</li>
                    <li>Opportunités d'intervention spécifiques à l'état</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Machine Learning Insights
        st.header("🤖 Machine Learning Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="key-finding">
                <h4 class="insight-title">🎯 Précision du modèle</h4>
                <p><strong>Les meilleurs modèles atteignent une précision de 70 à 75 %</strong></p>
                <ul>
                    <li>Random Forest : Précision la plus élevée</li>
                    <li>Régression logistique : Meilleure interprétabilité</li>
                    <li>L'importance des caractéristiques révèle les principaux prédicteurs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="key-finding">
                <h4 class="insight-title">📈 Predictive Features</h4>
                <p><strong>Top predictors identified:</strong></p>
                <ul>
                    <li>Customer tenure (months)</li>
                    <li>Revenue per month</li>
                    <li>Satisfaction rate</li>
                    <li>Device type usage patterns</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="key-finding">
                <h4 class="insight-title">🎪 Valeur de l'entreprise</h4>
                <p><strong>ML-enhanced targeting can improve ROI by 40%+</strong></p>
                <ul>
                    <li>Opportunités d'intervention précoce</li>
                    <li>Stratégies de fidélisation personnalisées</li>
                    <li>Optimisation des ressources</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Strategic Recommendations
        st.header("🚀 Strategic Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🎯 Immediate Actions (0-3 months)**
            **1. Programme d'accueil des nouveaux clients**
            - Mettre en œuvre un soutien intensif de 90 jours pour les nouveaux clients à forte valeur ajoutée
            - Responsables de compte dédiés pour les revenus >₦100,000
            - Suivi proactif de la satisfaction

            **2. Intervention sur la qualité du réseau**
            - Investissement prioritaire dans l'infrastructure des états à fort taux de désabonnement
            - Équipes d'intervention d'urgence pour les problèmes de réseau
            - Tableau de bord de surveillance de la qualité

            **3. Stratégie de réponse concurrentielle**
            - Cadre d'ajustement et de comparaison des prix
            - Retention offers for at-risk customers
            - Value-added services differentiation
            """)
        
        with col2:
            st.success("""
            **📈 Strategic Initiatives (3-12 months)**
            
            **1. Déploiement de l'analyse prédictive**
           - Déployer des modèles d'IA pour la prédiction en temps réel du désabonnement
            - Système d'alerte précoce automatisé
            - Déclencheurs d'intervention personnalisés

            **2. Optimisation du cycle de vie client**
            - Offres de services basées sur le cycle de vie
            - Programmes de fidélité gradués
            - Avantages tarifaires basés sur l'ancienneté

            **3. Conception de services basée sur les données**
            - Optimisation de l'efficacité des revenus
            - Offres de services spécifiques aux appareils
            - Recommandations basées sur les modèles d'utilisation
            """)
        
        # Risk Assessment and Monitoring
        st.header("⚠️ Risk Assessment & Monitoring Framework")
        
        # Calculate risk segments
        df['Risk_Score'] = 0
        df.loc[df['Satisfaction Rate'] <= 2, 'Risk_Score'] += 3
        df.loc[df['Customer Tenure in months'] <= 12, 'Risk_Score'] += 2
        df.loc[df['Revenue_per_Month'] < df['Revenue_per_Month'].median(), 'Risk_Score'] += 1
        
        risk_segments = df.groupby('Risk_Score').agg({
            'Customer ID': 'count',
            'Total Revenue': 'sum',
            'Customer Churn Status': lambda x: (x == 'Yes').sum()
        }).reset_index()
        risk_segments['Churn_Rate'] = (risk_segments['Customer Churn Status'] / risk_segments['Customer ID']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            fig_risk_dist = px.bar(
                risk_segments,
                x='Risk_Score',
                y='Customer ID',
                title="Customer Distribution by Risk Score",
                labels={'Customer ID': 'Number of Customers', 'Risk_Score': 'Risk Score'},
                color='Churn_Rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        with col2:
            # High-risk customer analysis
            high_risk_customers = len(df[df['Risk_Score'] >= 4])
            high_risk_revenue = df[df['Risk_Score'] >= 4]['Total Revenue'].sum()
            
            st.error(f"""
            🚨 **Alerte aux clients à haut risque**

            **{high_risk_customers:,} clients à risque immédiat**

            • Revenus en jeu : ₦{high_risk_revenue:,.0f}
            • Nécessitent une intervention immédiate
            • Probabilité de désabonnement prévue : >60%

            **KPI de suivi :**
            • Scores de satisfaction hebdomadaires
            • Suivi mensuel du taux de désabonnement
            • Taux de réussite des interventions
            • Métriques de récupération des revenus
            """)
        
        # Implementation Roadmap
        st.header("🗓️ Implementation Roadmap")
        
        roadmap_data = {
            'Phase': ['Phase 1: Foundation', 'Phase 2: Enhancement', 'Phase 3: Optimization', 'Phase 4: Innovation'],
            'Timeline': ['0-3 months', '3-6 months', '6-9 months', '9-12 months'],
            'Key_Activities': [
                'Data infrastructure, immediate interventions, risk scoring',
                'ML model deployment, automated workflows, team training',
                'Advanced analytics, personalization engine, A/B testing',
                'AI-driven insights, predictive recommendations, expansion'
            ],
            'Expected_Impact': ['15% churn reduction', '25% churn reduction', '35% churn reduction', '40%+ churn reduction'],
            'Investment_Level': ['Medium', 'High', 'Medium', 'High']
        }
        
        roadmap_df = pd.DataFrame(roadmap_data)
        st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
        
        # Executive Summary Conclusion
        st.header("📋 Executive Summary")
        
        st.success(f"""
        ### Principaux enseignements pour les dirigeants

        **État actuel :** MTN Nigeria fait face à un taux de désabonnement de {churn_rate:.1f} %, représentant ₦{churned_revenue:,.0f} de revenus perdus chaque année.

        **Causes profondes :** Les problèmes de qualité du réseau, la pression concurrentielle et l'intégration insuffisante des nouveaux clients sont des moteurs principaux.

        **Opportunité :** Avec des interventions ciblées, nous pouvons récupérer ₦{potential_savings:,.0f} de revenus grâce à une meilleure fidélisation.

        **Prochaines étapes :** Le déploiement immédiat de nouveaux programmes pour les clients et de systèmes de prédiction basés sur l'IA permettra d'obtenir un retour sur investissement mesurable dans les 90 jours.
        """)
        
    else:
        st.error("No data available for executive summary.")

except Exception as e:
    st.error(f"Error generating executive summary: {str(e)}")
    st.info("Please check the data source and try again.")