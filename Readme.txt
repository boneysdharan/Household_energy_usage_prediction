# ⚡ Household Energy Usage Dashboard

A **Streamlit-powered interactive dashboard** for analyzing and modeling **household energy consumption** using a real-world dataset.  
It features powerful **visual analytics**, **machine learning models**, and a **deep neural network** built with **PyTorch**.

> 📊 Ideal for students, data scientists, or engineers looking to explore energy usage patterns and predictive modeling with modern tools.

---

## 📁 Dataset
Data.txt contain dataset in .txt format
Data.csv contain dataset in .scv format
You can use any to do this porject(text is raw one and scv is clean data one)
- **Name**: [Individual household electric power consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Source**: UCI Machine Learning Repository
- **Size**: ~2 million rows
- **Period**: December 2006 to November 2010
- **Features**:
  - Global active power
  - Voltage
  - Global intensity
  - Sub-meter readings
  - Date and Time

---

## 🌟 Features

| Category               | Highlights                                                                 |
|------------------------|---------------------------------------------------------------------------|
| 📈 **EDA**             | Daily, weekly, and hourly usage trends                                     |
| 🧠 **ML Models**        | Linear Regression, Decision Tree, KNN, Random Forest, Deep Neural Network |
| 📊 **Metrics**          | R² Score, RMSE, MAE, residual analysis                                     |
| 🔍 **Feature Insights** | Feature importance plots for tree-based models                            |
| 📉 **Predictions**      | Actual vs Predicted graphs for each model                                 |
| ⚙️ **Customizable**      | Easily extendable and modular codebase                                    |

---

## 🛠️ Installation & Setup

### 🧾 Prerequisites

- Python 3.7+
- Recommended IDE: VS Code, PyCharm, Jupyter Lab

### 🔧 Install Required Packages

```bash
pip install -r requirements.txt
<details> <summary>📦 Click to view or create <code>requirements.txt</code></summary>
text
Copy
Edit
streamlit
pandas
numpy
scikit-learn
plotly
torch
seaborn
</details>
🚀 Launch the Dashboard
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/energy-usage-dashboard.git
cd energy-usage-dashboard
Add the dataset (household_power_consumption.csv) to the root folder.

Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
The dashboard will open in your default browser at http://localhost:8501.

📊 Dashboard Overview
1️⃣ Exploratory Data Analysis (EDA)
📅 Daily average power usage (line chart)

📆 Average usage by day of the week (bar chart)

⏰ Hourly usage trend (line chart)

📈 Global Active Power distribution (histogram)

2️⃣ Model Evaluation (Choose from):
Linear Regression

Decision Tree Regressor

K-Nearest Neighbors (KNN)

Random Forest Regressor

Deep Neural Network (PyTorch)

For each model:

📉 Actual vs Predicted plot

📐 Residuals scatter plot

📊 Performance metrics: R² Score, RMSE, MAE

🔍 Feature importance (if supported)

3️⃣ Model Comparison Table
Compares performance across all models based on:

Metric	Description
R² Score	Explained variance (higher = better)
RMSE	Root Mean Squared Error (lower = better)
MAE	Mean Absolute Error (lower = better)

🧠 Deep Learning Details
Implemented using PyTorch

Feedforward architecture with:

4 hidden layers: 64 → 32 → 16 → 8

ReLU activations

Optimizer: Adam

Loss function: Mean Squared Error (MSE)

Trained for 10 epochs on 200,000 sampled rows

📁 Project Structure
bash
Copy
Edit
energy-usage-dashboard/
│
├── app.py                          # Main Streamlit application
├── household_power_consumption.csv # Dataset (not included)
├── requirements.txt                # Python dependencies
├── assets/                         # (Optional) screenshots or images
└── README.md                       # You’re reading it!

!for individual analysis, refer to energy.ipynb file.
