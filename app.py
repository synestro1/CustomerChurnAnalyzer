import streamlit as st
import pandas as pd
from utils.data_processor import load_and_process_data

# Configure page
st.set_page_config(
    page_title="MTN Nigeria - Customer Churn Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for MTN branding
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #FFCC00 0%, #FFD700 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: #000000;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFCC00;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #FFF9E6;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #FFCC00;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>📱 MTN Nigeria - Customer Churn Analysis Dashboard</h1>
        <p>Analyse complète des schémas de fidélisation et d'attrition de la clientèle</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        # Load data
        df = load_and_process_data()

        if df is not None and not df.empty:
            # Aperçu des mesures
            st.header("📊 Key Metrics Overview")

            # Calculer les indicateurs clés
            total_customers = df["Customer ID"].nunique()
            total_churned = df[df["Customer Churn Status"] == "Yes"][
                "Customer ID"
            ].nunique()
            churn_rate = (total_churned / total_customers) * 100
            total_revenue = df["Total Revenue"].sum()
            avg_tenure = df["Customer Tenure in months"].mean()

            # Affichage des mesures en colonnes
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(label="Total Customers", value=f"{total_customers:,}")

            with col2:
                st.metric(label="Churned Customers", value=f"{total_churned:,}")

            with col3:
                st.metric(label="Churn Rate", value=f"{churn_rate:.1f}%")

            with col4:
                st.metric(label="Total Revenue", value=f"₦{total_revenue:,.0f}")

            with col5:
                st.metric(label="Avg Tenure", value=f"{avg_tenure:.1f} months")

            # Aperçu rapide
            st.header("🎯 Quick Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                <div class="insight-box">
                    <h4>🔍 Churn Analysis</h4>
                    <p>Mon analyse révèle des schémas clés dans le comportement de désabonnement des clients, ce qui permet d'identifier les segments à risque et les opportunités de fidélisation.</p>
                    <ul>
                        <li>Il existe une forte Lien entre la satisfaction du client et la fidélisation</li>
                        <li>Les problèmes de qualité du réseau sont un facteur important de désabonnement</li>
                        <li>Les offres compétitives posent d'importants défis en matière de fidélisation</li>
                    </ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
                <div class="insight-box">
                    <h4>📈 Business Impact</h4>
                    <p>La compréhension des schémas de désabonnement permet de mettre en place des stratégies proactives de fidélisation de la clientèle et de protection des revenus.</p>
                    <ul>
                        <li>Impact sur le chiffre d'affaires de la perte de clients</li>
                        <li>Stratégies de fidélisation spécifiques à un segment</li>
                        <li>Indicateurs prédictifs pour une intervention précoce</li>
                    </ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Intégration analytique avancée
            st.header("🧠 Advanced Analytics Features")

            # Calculer le revenu par mois pour des mesures améliorées
            df["Revenue_per_Month"] = df.apply(
                lambda row: (
                    row["Total Revenue"] / row["Customer Tenure in months"]
                    if row["Customer Tenure in months"] > 0
                    else 0
                ),
                axis=1,
            )

            # Amélioration de l'affichage des mesures
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_rev_churned = df[df["Customer Churn Status"] == "Yes"][
                    "Revenue_per_Month"
                ].mean()
                avg_rev_retained = df[df["Customer Churn Status"] == "No"][
                    "Revenue_per_Month"
                ].mean()
                revenue_gap = avg_rev_retained - avg_rev_churned
                st.metric(
                    "Monthly Revenue Gap",
                    f"₦{revenue_gap:,.0f}",
                    "Retained vs Churned customers",
                )

            with col2:
                # Evaluation des risques
                df["Risk_Score"] = 0
                df.loc[df["Satisfaction Rate"] <= 2, "Risk_Score"] += 3
                df.loc[df["Customer Tenure in months"] <= 12, "Risk_Score"] += 2
                df.loc[
                    df["Revenue_per_Month"] < df["Revenue_per_Month"].median(),
                    "Risk_Score",
                ] += 1

                high_risk_customers = len(df[df["Risk_Score"] >= 4])
                st.metric(
                    "High-Risk Customers",
                    f"{high_risk_customers:,}",
                    "Risk Score ≥ 4 (Immediate attention needed)",
                )

            with col3:
                # Préparation à la prédiction ML
                feature_completeness = (df.notna().sum() / len(df) * 100).mean()
                st.metric(
                    "Data Quality Score",
                    f"{feature_completeness:.1f}%",
                    "Ready for ML deployment",
                )

            # Guide de navigation avec capacités de ML
            st.header("🧭 Complete Analytics Suite")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.info(
                    """
                **📊 Advanced Insights**
                
                intelligence Business avec analyse du cycle de vie des clients, études d'impact sur le chiffre d'affaires et recommandations stratégiques.
                """
                )

            with col2:
                st.info(
                    """
                **📈 Enhanced Visualizations**
                
                Tableaux de bord interactifs avec analyse de l'efficacité des revenus, évaluation des risques et indicateurs prédictifs.
                """
                )

            with col3:
                st.info(
                    """
                **📋 Data Explorer**
                
                Filtrage complet, recherche de clients, analyses détaillées et possibilités d'exportation.
                """
                )

            with col4:
                st.info(
                    """
                **🤖 ML Models**
                
                Machine learning predictions, l'analyse de l'importance des caractéristiques et les simulations de l'impact sur les entreprises.
                """
                )

            with col5:
                st.info(
                    """
                **📋 Executive Summary**
                
                Vue d'ensemble stratégique avec les principales conclusions, la feuille de route de mise en œuvre et les recommandations des dirigeants.
                """
                )

            # Data aperçu
            st.header("📋 Data Preview")
            st.write("Échantillon de l'ensemble de données sur le taux de fidélisation des clients:")
            st.dataframe(df.head(10), use_container_width=True)

        else:
            st.error("No data available. Please check the data file and try again.")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the data file is properly formatted and accessible.")


if __name__ == "__main__":
    main()
