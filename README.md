# 🏠 PropWise AI

PropWise AI is an advanced, end-to-end Machine Learning web application designed to revolutionize how property evaluations are conducted. By combining cutting-edge predictive algorithms with an Agentic LLM layer (Groq), PropWise AI delivers high-precision market valuations alongside structured, AI-generated real estate advisory reports.

---

## ✨ Key Features

1. **🚀 High-Precision ML Valuations (`Gradient Boosting` / `XGBoost`)**
   - PropWise sweeps multiple algorithms (`RandomForest`, `GradientBoosting`, and `XGBoost`) automatically tuning hyperparameters via `GridSearchCV` to deploy the most accurate property valuation model.
   
2. **🤖 Agentic AI Advisory Engine (`Groq` / `Llama-3.3-70b`)**
   - Going beyond simple predictions, PropWise utilizes an integrated LLM to interpret the predicted price and property data, returning a formally structured, cautious advisory report (including market trends, interpretation, and recommended next steps).

3. **📊 Interactive Streamlit Dashboard**
   - **Predict Price:** Instant single-property valuations and Batch-CSV property inference.
   - **Comparable Properties:** An interactive engine that searches the datastore for the 5 most physically similar properties, predicts their values, and graphs how your property compares to the neighborhood average.
   - **Model Performance Tracker:** A dedicated observability layer that actively tracks MSE, RMSE, R² scores, and visualizes Feature Importance.
   
4. **📄 Automated PDF Export**
   - Downloads customized, dynamically generated PDF dossiers containing the property snapshot, the Agent's insights, and necessary legal disclaimers natively via `fpdf2`.

---

## 🏗️ Architecture

```bash
📦 PropWise-AI
 ┣ 📂 data                 # Housing.csv dataset
 ┣ 📜 app.py               # The main Streamlit Front-End 
 ┣ 📜 agent.py             # Agentic Layer interfacing with Groq API
 ┣ 📜 analyze_housing.py   # Sklearn/XGBoost Multi-Model Training Engine
 ┣ 📜 requirements.txt     # Python dependencies
 ┣ 📜 .env                 # Environment variables (API Keys)
 ┗ ⚙️ model.pkl            # Pre-compiled optimized ML Pipeline
```

---

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone https://github.com/yourusername/PropWise-AI.git
cd PropWise-AI

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
To enable the **AI Advisory** features, you must provide a free [Groq API Key](https://console.groq.com/keys). 
Create a file strictly named `.env` in the root folder with the following contents:

```env
# Get your FREE Groq API key at: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Model Training (Optional but Recommended)
The repository automatically attempts to retrain if a `model.pkl` mismatch is detected. To manually train the model on your environment's `scikit-learn` version:

```bash
python analyze_housing.py
```
*This will perform a grid search sweep and generate heavily tuned `model.pkl`, `metrics.json`, and `feature_importance.csv` artifacts.*

### 4. Running the Dashboard
Boot up the Streamlit interface:

```bash
streamlit run app.py
```
Once initialized, navigate to `http://localhost:8501` in your browser.

---

## ⚠️ Disclaimer
PropWise AI and its generated advisory reports utilize predictive logic and Generative AI. These outputs are created strictly for **indicative and demonstration purposes** and **do not constitute financial, legal, or investment advice**. Always consult a licensed local real estate professional for transactional decisions.