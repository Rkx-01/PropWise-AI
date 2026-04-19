import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import subprocess
import json
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load .env (GROQ_API_KEY) if present — no-op when already in environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import generate_advisory_report

# Page configuration
st.set_page_config(layout="wide", page_title="PropWise AI - Property Price Predictor")

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Artifact Loading
def load_artifacts():
    try:
        pipeline = joblib.load('model.pkl')
    except Exception as e:
        st.warning(f"Model mismatch detected ({e}). Rebuilding the model locally...")
        # Automatically retrain if the pickle is incompatible
        import sys
        import subprocess
        subprocess.run([sys.executable, "analyze_housing.py"], check=True)
        try:
            result = subprocess.run([sys.executable, "analyze_housing.py"], capture_output=True, text=True, check=True)
            st.success("Model rebuilt successfully.")
            pipeline = joblib.load('model.pkl')
        except subprocess.CalledProcessError as e_build:
            st.error(f"Failed to rebuild the model. Error: {e_build.stderr}")
            return None, {}, None
        except Exception as e2:
            st.error(f"Failed to load rebuilt model: {e2}")
            return None, {}, None

    try:
        metrics = {}
        if os.path.exists('metrics.json'):
            with open('metrics.json', 'r') as f:
                metrics = json.load(f)
        df_imp = pd.read_csv('feature_importance.csv') if os.path.exists('feature_importance.csv') else None
        return pipeline, metrics, df_imp
    except Exception as e:
        st.error(f"Error loading secondary artifacts: {e}")
        return pipeline, {}, None

# Feature Engineering
def engineer_features(df):
    current_year = datetime.now().year
    if 'year_built' not in df.columns:
        df['year_built'] = current_year 
    df['property_age'] = current_year - df['year_built']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    return df

# --- PAGE: Home ---
def render_home():
    st.title("🏠 PropWise AI: Property Price Predictor")
    st.markdown("---")
    st.markdown("""
    ### Welcome to PropWise AI
    PropWise AI is an advanced machine learning dashboard designed to revolutionize how property evaluations are conducted. By combining spatial attributes, structural features, and modern amenities, our system delivers high-precision market valuations in seconds.
    
    #### 📂 Dataset Highlights
    The engine is trained on the comprehensive Housing dataset, focusing on:
    - **Spatial Characteristics**: Total area (sq ft), number of stories, and parking capacity.
    - **Structural Details**: Room distribution (bedrooms and bathrooms).
    - **Premium Features**: Air conditioning, preferred area status, and accessibility.
    - **Refurbishment Status**: Current furnishing condition.
    
    #### ⚙️ The Pipeline Workflow
    1. **Data Ingestion**: Loading raw historical housing data.
    2. **Transformation**: Automated feature engineering.
    3. **Preprocessing**: One-Hot Encoding and Scaling.
    4. **Modeling**: Random Forest Regressor Pipeline.
    
    #### 🚀 How to use
    1. **Explore Data**: Navigate to **Data Explorer** to view the dataset structure and correlation heatmaps.
    2. **Get Valuations**: Go to **Predict Price** to enter property details for an instant valuation.
    3. **Batch Processing**: Upload a CSV file via the **Sidebar** or the Predict page to process multiple properties at once.
    4. **Analyze Performance**: Check **Model Performance** to understand the accuracy and key factors driving property prices.
    """)

# --- PAGE: Data Explorer ---
def render_data_explorer():
    st.title("🔍 Data Explorer")
    st.markdown("---")
    
    data_path = os.path.join("data", "Housing.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📋 Dataset Overview")
            st.write(f"**Total Properties:** {df.shape[0]}")
            st.write(f"**Total Features:** {df.shape[1]}")
            st.dataframe(df.head(5), use_container_width=True)
            
        with col2:
            st.subheader("📈 Quick Statistics")
            st.dataframe(df.describe().T, use_container_width=True)
            
        st.markdown("---")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("🔥 Feature Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            plt.tight_layout()
            st.pyplot(fig_corr)
            
        with col_viz2:
            st.subheader("💰 Price Distribution Analysis")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 8))
            sns.histplot(df['price'], kde=True, color='#007bff', ax=ax_dist)
            ax_dist.set_title("Property Price Distribution")
            ax_dist.set_xlabel("Price")
            plt.tight_layout()
            st.pyplot(fig_dist)
    else:
        st.warning("Housing dataset not found. Please ensure `data/Housing.csv` is correctly placed.")

# --- PAGE: Predict Price ---
def render_predict_price(pipeline, sidebar_file=None):
    st.title("📍 Predict Property Price")
    st.markdown("---")
    
    if pipeline is None:
        st.error("Model artifacts not found. Please ensure training has been completed.")
        return

    tab1, tab2 = st.tabs(["Single Property valuation", "Batch Batch Prediction (Optional)"])
    
    with tab1:
        st.subheader("Property Specification Form")
        with st.form("valuation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                area = st.number_input("Total Area (sq ft)", value=5000, step=100)
                bedrooms = st.number_input("Number of Bedrooms", value=3, min_value=1, max_value=10)
                bathrooms = st.number_input("Number of Bathrooms", value=2, min_value=1, max_value=5)
                stories = st.number_input("Total Stories", value=2, min_value=1, max_value=4)
                parking = st.number_input("Parking Capacity", value=1, min_value=0, max_value=3)
                furnishingstatus = st.selectbox("Current Furnishing", ["furnished", "semi-furnished", "unfurnished"])
                
            with col2:
                mainroad = st.selectbox("Main Road Access", ["yes", "no"])
                guestroom = st.selectbox("Guestroom Availability", ["yes", "no"])
                basement = st.selectbox("Basement Level", ["yes", "no"])
                airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
                prefarea = st.selectbox("Preferred Location", ["yes", "no"])
                hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
            
            st.markdown(" ")
            submit = st.form_submit_button("💰 Get Instant Valuation")
            
        if submit:
            input_dict = {
                'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 
                'stories': stories, 'mainroad': mainroad, 'guestroom': guestroom,
                'basement': basement, 'airconditioning': airconditioning, 
                'parking': parking, 'prefarea': prefarea, 'furnishingstatus': furnishingstatus,
                'hotwaterheating': hotwaterheating, 'year_built': datetime.now().year
            }
            input_df = engineer_features(pd.DataFrame([input_dict]))
            prediction = pipeline.predict(input_df)[0]
            
            st.success(f"### Estimated Market Valuation: ₹{prediction:,.2f}")
            st.markdown("---")
            st.info("This prediction is generated based on current model training on historical trends.")

    with tab2:
        st.subheader("📋 Batch Processing Pipeline")
        
        # Determine which file to use
        current_file = sidebar_file if sidebar_file is not None else st.file_uploader("Upload CSV property list", type=["csv"], key="batch_uploader")
        
        if current_file:
            df_input = pd.read_csv(current_file)
            st.info(f"Loaded {len(df_input)} properties for analysis.")
            st.dataframe(df_input.head(10), use_container_width=True)
            
            if st.button("🚀 Process Batch valuation"):
                X_batch = engineer_features(df_input.copy())
                results = df_input.copy()
                results['Predicted Price'] = pipeline.predict(X_batch)
                results['Formatted Price'] = results['Predicted Price'].apply(lambda x: f"₹{x:,.2f}")
                
                st.subheader("✅ Processed Results")
                st.dataframe(results, use_container_width=True)
                
                csv_data = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download All Predictions", 
                    csv_data, 
                    "property_valuations.csv", 
                    "text/csv"
                )
        else:
            st.info("Please upload a CSV file via the sidebar or the box above to begin batch processing.")

# ---------------------------------------------------------------------------
# PDF Generator
# ---------------------------------------------------------------------------

def _generate_pdf(report: dict, predicted_price: float, property_data: dict) -> bytes:
    """Builds a clean PDF advisory report and returns raw bytes."""
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(30, 64, 175)          # indigo
            self.cell(0, 10, "PropWise AI - Advisory Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
                      align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(4)
            self.set_draw_color(30, 64, 175)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def safe(text):
        """Strip non-latin1 chars that basic PDF fonts can't handle."""
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(14, 14, 14)

    # ── Predicted price banner ─────────────────────────────────────────────
    pdf.set_fill_color(239, 246, 255)
    pdf.set_draw_color(147, 197, 253)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 12, safe(f"  Predicted Market Value:  Rs {predicted_price:,.0f}"),
             border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Property snapshot ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Property Snapshot", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    cols = list(property_data.items())
    for i in range(0, len(cols), 2):
        k1, v1 = cols[i]
        left  = safe(f"{k1.replace('_',' ').title()}: {v1}")
        right = ""
        if i + 1 < len(cols):
            k2, v2 = cols[i + 1]
            right = safe(f"{k2.replace('_',' ').title()}: {v2}")
        pdf.cell(90, 6, left)
        pdf.cell(90, 6, right, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Report sections ────────────────────────────────────────────────────
    sections = [
        ("Property Summary",      report.get("property_summary", "-")),
        ("Price Interpretation",   report.get("price_interpretation", "-")),
        ("Market Trend Insights",  report.get("market_trend_insights", "-")),
        ("Recommended Actions",    report.get("recommended_actions", "-")),
    ]
    for title, body in sections:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 64, 175)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, safe(body))
        pdf.ln(3)

    # Supporting references
    refs = report.get("supporting_references", [])
    if refs:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 64, 175)
        pdf.cell(0, 8, "Supporting References", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        for i, ref in enumerate(refs, 1):
            pdf.multi_cell(0, 6, safe(f"{i}. {ref}"))
        pdf.ln(3)

    # Legal disclaimer box
    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(251, 191, 36)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(120, 53, 15)
    pdf.cell(0, 7, "  Legal Disclaimer", border="TLR", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, safe("  " + report.get("legal_disclaimer", "")), border="BLR", fill=True)

    return bytes(pdf.output())


# --- PAGE: AI Advisory Report ---
def render_advisory(pipeline):
    st.title("🤖 AI Advisory Report")
    st.markdown("---")
    st.markdown(
        "Enter property details below, then click **Generate Advisory Report** "
        "to receive a structured real-estate advisory powered by Groq LLM."
    )

    if pipeline is None:
        st.error("Model not loaded — please run `python analyze_housing.py` first.")
        return

    # ── Property input form ────────────────────────────────────────────────
    with st.form("advisory_form"):
        st.subheader("🏠 Property Specification")
        col1, col2 = st.columns(2)
        with col1:
            area             = st.number_input("Total Area (sq ft)",  value=5000, step=100)
            bedrooms         = st.number_input("Bedrooms",             value=3,    min_value=1, max_value=10)
            bathrooms        = st.number_input("Bathrooms",            value=2,    min_value=1, max_value=5)
            stories          = st.number_input("Stories",              value=2,    min_value=1, max_value=4)
            parking          = st.number_input("Parking Spots",        value=1,    min_value=0, max_value=3)
            furnishingstatus = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])
        with col2:
            mainroad         = st.selectbox("Main Road Access",  ["yes", "no"])
            guestroom        = st.selectbox("Guest Room",         ["yes", "no"])
            basement         = st.selectbox("Basement",           ["yes", "no"])
            airconditioning  = st.selectbox("Air Conditioning",   ["yes", "no"])
            prefarea         = st.selectbox("Preferred Location", ["yes", "no"])
            hotwaterheating  = st.selectbox("Hot Water Heating",  ["yes", "no"])

        submit_advisory = st.form_submit_button(
            "🧠 Generate Advisory Report",
            use_container_width=True,
        )

    if not submit_advisory:
        return

    # ── Predict price ──────────────────────────────────────────────────────
    input_dict = {
        'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'stories': stories, 'mainroad': mainroad, 'guestroom': guestroom,
        'basement': basement, 'airconditioning': airconditioning,
        'parking': parking, 'prefarea': prefarea,
        'furnishingstatus': furnishingstatus,
        'hotwaterheating': hotwaterheating,
        'year_built': datetime.now().year,
    }
    input_df        = engineer_features(pd.DataFrame([input_dict]))
    predicted_price = float(pipeline.predict(input_df)[0])

    # st.metric banner
    st.markdown("### 📊 ML Price Estimate")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Price",       f"₹{predicted_price:,.0f}")
    m2.metric("Est. Per sq ft",        f"₹{predicted_price / max(area, 1):,.0f}")
    m3.metric("Confidence Band (±15%)",
              f"₹{predicted_price*0.85:,.0f} – ₹{predicted_price*1.15:,.0f}")
    st.markdown("---")

    # ── Build agent property dict ──────────────────────────────────────────
    property_data = {
        **input_dict,
        'property_age': datetime.now().year - input_dict['year_built'],
        'total_rooms' : bedrooms + bathrooms,
    }
    property_data.pop('year_built', None)

    # ── Call agent ─────────────────────────────────────────────────────────
    with st.spinner("Consulting Groq LLM advisor … this is usually very fast!"):
        report = generate_advisory_report(property_data, predicted_price)

    if "error" in report:
        st.error(f"⚠️ Agent error: {report['error']}")

    # ── Render report with expanders ───────────────────────────────────────
    st.markdown("### 📑 Advisory Report")

    with st.expander("📋 Property Summary", expanded=True):
        st.write(report.get("property_summary", "—"))

    with st.expander("💰 Price Interpretation", expanded=True):
        st.write(report.get("price_interpretation", "—"))

    with st.expander("📈 Market Trend Insights", expanded=True):
        st.write(report.get("market_trend_insights", "—"))

    with st.expander("✅ Recommended Actions", expanded=True):
        st.write(report.get("recommended_actions", "—"))

    refs = report.get("supporting_references", [])
    if refs:
        with st.expander("📚 Supporting References", expanded=False):
            for i, ref in enumerate(refs, 1):
                st.markdown(f"{i}. {ref}")

    # ── Legal disclaimer — always visible ──────────────────────────────────
    st.warning(
        "⚖️ **Legal Disclaimer** — "
        + report.get("legal_disclaimer",
                     "This report is AI-generated and does not constitute financial, "
                     "legal, or investment advice. Consult a licensed professional.")
    )

    # ── PDF Download ────────────────────────────────────────────────────────
    st.markdown("---")
    try:
        pdf_bytes = _generate_pdf(report, predicted_price, property_data)
        st.download_button(
            label="📄 Download Report as PDF",
            data=pdf_bytes,
            file_name=f"propwise_advisory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as pdf_err:
        st.warning(f"PDF export failed: {pdf_err}")

    # ── Raw JSON (dev expander) ─────────────────────────────────────────────
    with st.expander("🔍 Raw JSON Report"):
        st.json(report)


# --- PAGE: Model Performance ---
def render_model_performance(metrics, df_imp):
    st.title("📈 Model Performance Monitoring")
    st.markdown("---")
    
    if metrics:
        if isinstance(metrics, list) and len(metrics) > 0:
            metrics = metrics[0]  # Grab the best baseline model's metrics
            
        st.subheader("🎯 Key Performance Indicators")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Mean Absolute Error (MAE)", f"₹{metrics['MAE']:,.0f}")
        with m_col2:
            st.metric("Root Mean Squared Error (RMSE)", f"₹{metrics['RMSE']:,.0f}")
        with m_col3:
            st.metric("R2 Variance Score", f"{metrics['R2']:.4f}")
    else:
        st.warning("Performance metrics file `metrics.json` missing.")
        
    st.markdown("---")
    
    if df_imp is not None:
        st.subheader("🔍 Top Price-Driving Factors")
        st.markdown("These features have the highest relative impact on the property valuation.")
        
        top_10 = df_imp.head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, 10))
        
        ax.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color=colors)
        ax.set_xlabel('Relative Importance')
        ax.set_title('Top 10 Feature Importance (Random Forest)')
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("Explore Full Feature Statistics"):
            st.dataframe(df_imp, use_container_width=True)
    else:
        st.warning("Feature importance data missing.")

# --- PAGE: Comparable Properties ---
def render_comparables(pipeline):
    st.title("🏡 Comparable Properties")
    st.markdown("---")
    st.markdown("Find the top 5 most similar properties in the dataset and compare their predicted vs. actual prices to your property's valuation.")
    
    if pipeline is None:
        st.error("Model not loaded — please run `python analyze_housing.py` first.")
        return
        
    data_path = os.path.join("data", "Housing.csv")
    if not os.path.exists(data_path):
        st.warning("Housing dataset not found.")
        return
        
    with st.form("comparable_form"):
        st.subheader("Your Property Details")
        
        col1, col2 = st.columns(2)
        with col1:
            area             = st.number_input("Total Area (sq ft)",  value=5000, step=100)
            bedrooms         = st.number_input("Bedrooms",             value=3,    min_value=1, max_value=10)
            bathrooms        = st.number_input("Bathrooms",            value=2,    min_value=1, max_value=5)
            stories          = st.number_input("Stories",              value=2,    min_value=1, max_value=4)
            parking          = st.number_input("Parking Spots",        value=1,    min_value=0, max_value=3)
            furnishingstatus = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])
        with col2:
            mainroad         = st.selectbox("Main Road Access",  ["yes", "no"])
            guestroom        = st.selectbox("Guest Room",         ["yes", "no"])
            basement         = st.selectbox("Basement",           ["yes", "no"])
            airconditioning  = st.selectbox("Air Conditioning",   ["yes", "no"])
            prefarea         = st.selectbox("Preferred Location", ["yes", "no"])
            hotwaterheating  = st.selectbox("Hot Water Heating",  ["yes", "no"])

        submit = st.form_submit_button("🔍 Find Comparables")

    if submit:
        # Load dataset
        df_all = pd.read_csv(data_path)
        
        # 1. Filter dataset: area (±20%), bedrooms (exact), furnishingstatus (exact)
        min_area = area * 0.8
        max_area = area * 1.2
        
        df_filtered = df_all[
            (df_all['area'] >= min_area) &
            (df_all['area'] <= max_area) &
            (df_all['bedrooms'] == bedrooms) &
            (df_all['furnishingstatus'] == furnishingstatus)
        ].copy()
        
        if len(df_filtered) == 0:
            st.warning("No comparable properties found within ±20% area, exact bedrooms, and furnishing status.")
            return
            
        # Top 5 closest in area
        df_filtered['area_diff'] = abs(df_filtered['area'] - area)
        df_top5 = df_filtered.sort_values(by='area_diff').head(5).drop(columns=['area_diff'])
        
        # 2. Predict for the user's property 
        user_dict = {
            'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
            'stories': stories, 'mainroad': mainroad, 'guestroom': guestroom,
            'basement': basement, 'airconditioning': airconditioning,
            'parking': parking, 'prefarea': prefarea,
            'furnishingstatus': furnishingstatus,
            'hotwaterheating': hotwaterheating,
            'year_built': datetime.now().year,
        }
        user_df = engineer_features(pd.DataFrame([user_dict]))
        user_predicted_price = float(pipeline.predict(user_df)[0])
        
        # 3. Predict prices for Top 5
        df_top5_feat = df_top5.copy()
        df_top5_feat = engineer_features(df_top5_feat) 
        
        X_top5 = df_top5_feat.drop(columns=['price'])
        if 'price_per_sqft' in X_top5.columns:
            X_top5 = X_top5.drop(columns=['price_per_sqft'])
            
        df_top5['predicted_price'] = pipeline.predict(X_top5)
        
        # 4. Display comparison table
        display_df = df_top5[['area', 'bedrooms', 'bathrooms', 'furnishingstatus', 'price', 'predicted_price']].copy()
        display_df = display_df.rename(columns={'price': 'actual_price'})
        
        st.subheader("📋 Top 5 Comparable Properties")
        formatted_df = display_df.copy()
        formatted_df['actual_price'] = formatted_df['actual_price'].apply(lambda x: f"₹{x:,.0f}")
        formatted_df['predicted_price'] = formatted_df['predicted_price'].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(formatted_df, use_container_width=True)
        
        # 5. Bar chart comparing user's property with comparables
        st.subheader("📈 Predicted Price Comparison")
        
        names = ["Your Property"] + [f"Comp {i+1}" for i in range(len(df_top5))]
        prices = [user_predicted_price] + df_top5['predicted_price'].tolist()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, prices, color=['#ff9999'] + ['#99ccff'] * len(df_top5))
        ax.set_ylabel("Predicted Price (₹)")
        ax.set_title("Your Property vs. Closest Comparables")
        
        # Add labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"₹{height:,.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig)
        
        # 6. Summary line
        avg_comp = df_top5['predicted_price'].mean()
        if avg_comp > 0:
            diff = user_predicted_price - avg_comp
            pct_diff = (diff / avg_comp) * 100
            direction = "above" if pct_diff > 0 else "below"
            st.info(f"💡 **Summary:** Your property is priced **{abs(pct_diff):.1f}% {direction}** the average of comparable properties (*Avg: ₹{avg_comp:,.0f}*).")

# --- MAIN EXECUTION ---
def main():
    # Load persistence
    pipeline, metrics, df_imp = load_artifacts()

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("PropWise AI Dashboard")
    st.sidebar.markdown("*Empowering property intelligence*")
    st.sidebar.markdown("---")

    navigation = st.sidebar.radio(
        "Navigation Menu",
        ["Home", "Data Explorer", "Predict Price", "Comparable Properties", "Model Performance", "AI Advisory Report"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Data for Prediction")
    sidebar_file = st.sidebar.file_uploader("Upload CSV for Batch Prediction", type=["csv"])

    st.sidebar.markdown("---")

    # ── Page dispatcher ────────────────────────────────────────────────────
    if navigation == "Home":
        render_home()
    elif navigation == "Data Explorer":
        render_data_explorer()
    elif navigation == "Predict Price":
        render_predict_price(pipeline, sidebar_file)
    elif navigation == "Comparable Properties":
        render_comparables(pipeline)
    elif navigation == "Model Performance":
        render_model_performance(metrics, df_imp)
    elif navigation == "AI Advisory Report":
        render_advisory(pipeline)


if __name__ == "__main__":
    main()
