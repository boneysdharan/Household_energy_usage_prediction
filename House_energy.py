import streamlit as st
import pandas as pd

import plotly.express as px

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Streamlit page config
st.set_page_config(page_title="Energy Usage Dashboard", layout="wide")

# --- Load and preprocess data ---

def load_data():
    path = r"D:\BONEYS\WEB\PYTHON\Project\HouseHold_energy_usage\household_power_consumption.csv"
    df = pd.read_csv(path,low_memory=False)

    df = df.ffill()
    df = df.bfill()

    # Combine 'Date' and 'Time' into a single datetime column
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')

    # Convert Global_active_power to numeric (handle non-numeric values)
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'],errors='coerce')

    # Set DateTime as index
    df.set_index('DateTime', inplace=True)

    # Daily Average
    df['Daily_Average'] = df['Global_active_power'].resample('D').transform('mean')

    # Peak Hour during the day
    df['Hour'] = df.index.hour
    daily_peak_hours = (
        df.groupby(['Date', 'Hour'])['Global_active_power']
        .sum()
        .reset_index()
        .sort_values(['Date', 'Global_active_power'], ascending=[True, False])
        .drop_duplicates(subset='Date')  # keep only the top hour per date
        .rename(columns={'Hour': 'Peak_Hour'})
    )

    if 'Peak_Hour' in df.columns:
        df = df.drop(columns=['Peak_Hour'])

    # merge to get Peak Hour during the day
    df.reset_index(inplace=True)  # brings DateTime back as a column
    df = df.merge(daily_peak_hours[['Date', 'Peak_Hour']], on='Date', how='left')
    df.set_index('DateTime', inplace=True)

    df = df.drop(columns=['Hour'],axis=1)
    # Convert object columns to numeric, excluding 'Date' and 'Time'
    cols_to_convert = df.select_dtypes(include='object').columns.drop(['Date', 'Time'])

    # Apply conversion
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    #removing outliners we have 2075259 rows
    df['Voltage_normalized'] = (df['Voltage']-df['Voltage'].mean())/df['Voltage'].std()
    df['Global_intensity_normalized'] = (df['Global_intensity']-df['Global_intensity'].mean())/df['Global_intensity'].std()
    condition1 = df['Voltage_normalized']<=3
    condition2 = df['Voltage_normalized']>=-3
    condition3 = df['Global_intensity_normalized']<=3
    condition4 = df['Global_intensity_normalized']>=-3
    df = df[(condition1 & condition2 & condition3 & condition4)]

    df = df.sample(n=min(200000, len(df)), random_state=42)
    return df

df = load_data()

# --- UI Header ---
st.title("📊 Household Energy Usage Dashboard")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 EDA", "🤖 Model Evaluation","🤖 Model Metrics Comparision"])

# -------------------------------
# 📈 Tab 1: Exploratory Data Analysis
# -------------------------------
with tab1:
    st.subheader("Average Global Active Power Over Time")
    daily_power = df['Global_active_power'].resample('D').mean()
    fig = px.line(daily_power.reset_index(), x='DateTime', y='Global_active_power', 
                labels={'Global_active_power': 'Power (kW)', 'DateTime': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔄 Average Power Usage by Day of Week")
    df['DayOfWeek'] = df.index.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = df.groupby('DayOfWeek')['Global_active_power'].mean().reindex(day_order).reset_index()

    fig_dow = px.bar(
        dow_avg,
        x='DayOfWeek',
        y='Global_active_power',
        labels={'Global_active_power': 'Average Power (kW)', 'DayOfWeek': 'Day'},
        color='Global_active_power',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    st.subheader("🕓 Average Power Usage by Hour of the Day")
    df['Hour'] = df.index.hour
    hourly_avg = df.groupby('Hour')['Global_active_power'].mean().reset_index()

    fig_hour = px.line(
        hourly_avg,
        x='Hour',
        y='Global_active_power',
        labels={'Global_active_power': 'Average Power (kW)', 'Hour': 'Hour of Day'},
        markers=True
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("📊 Distribution of Global Active Power")
    fig_hist = px.histogram(
        df,
        x='Global_active_power',
        nbins=100,
        labels={'Global_active_power': 'Power (kW)'},
        color_discrete_sequence=['purple']
    )
    fig_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------
# 🤖 Tab 2: Model Evaluation
# -------------------------------
with tab2:
    st.subheader("📌 Model-Based Evaluation (Select a Model)")

    model_choice = st.selectbox("Choose Model to Evaluate", [
        "Linear Regression", "Random Forest", "Decision Tree", "K-Nearest Neighbors", "Deep Neural Network"
    ])

    df_sample = df.sample(n=min(200000, len(df)), random_state=42)
    X = df_sample.drop(columns=['Date', 'Time', 'Global_active_power', 'Peak_Hour', 
                                'Voltage_normalized', 'Global_intensity_normalized','DayOfWeek'])
    y = df_sample['Global_active_power']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2,random_state=42) #makes sure we use the same 2,00,000 dataset

    # ---------- Train selected model ----------
    if model_choice == "Deep Neural Network":
        xtrain_tensor = torch.tensor(xtrain.values, dtype=torch.float32)
        xtest_tensor = torch.tensor(xtest.values, dtype=torch.float32)
        ytrain_tensor = torch.tensor(ytrain.values, dtype=torch.float32)
        ytest_tensor = torch.tensor(ytest.values, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(xtrain_tensor, ytrain_tensor), batch_size=128, shuffle=True)

        class DeepNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(DeepNN, self).__init__()
                self.input = nn.Linear(input_dim, 64)
                self.hidden1 = nn.Linear(64, 32)
                self.hidden2 = nn.Linear(32, 16)
                self.hidden3 = nn.Linear(16, 8)
                self.output = nn.Linear(8, output_dim)

            def forward(self, x):
                x = torch.relu(self.input(x))
                x = torch.relu(self.hidden1(x))
                x = torch.relu(self.hidden2(x))
                x = torch.relu(self.hidden3(x))
                return self.output(x)

        model = DeepNN(xtrain_tensor.shape[1], 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            for xb, yb in train_loader:
                yb = yb.view(-1, 1)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds = model(xtest_tensor).squeeze().numpy()
        ytrue = ytest_tensor.numpy()

    else:
        model_map = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Decision Tree": DecisionTreeRegressor(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
        }

        model = model_map[model_choice]
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        ytrue = ytest.values

    # ---------- Visualizations ----------
    import plotly.graph_objects as go
    st.subheader("📉 Actual vs Predicted (First 2000 Samples)")
    plot_df = pd.DataFrame({
        'Sample': list(range(2000)),
        'Actual': ytrue[:2000],
        'Predicted': preds[:2000]
    })
    fig4 = go.Figure()
    # Actual values line
    fig4.add_trace(go.Scatter(
        x=plot_df['Sample'],
        y=plot_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2),
        hovertemplate='Sample: %{x}<br>Actual: %{y:.3f}<extra></extra>'
    ))
    # Predicted values line
    fig4.add_trace(go.Scatter(
        x=plot_df['Sample'],
        y=plot_df['Predicted'],
        mode='lines',
        name='Predicted',
        line=dict(color='orange', width=2, dash='dot'),
        hovertemplate='Sample: %{x}<br>Predicted: %{y:.3f}<extra></extra>'
    ))
    # Layout improvements
    fig4.update_layout(
        xaxis_title="Sample Index",
        yaxis_title="Global Active Power (kW)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ---------- Metrics ----------
    rmse = root_mean_squared_error(ytrue, preds)
    mae = mean_absolute_error(ytrue, preds)
    r2 = r2_score(ytrue, preds)

    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("MAE", f"{mae:.4f}")

# ---------- Residuals Scatter Plot ----------
    st.subheader("🔍 Residuals Scatter Plot")
    residuals = ytrue - preds

    fig_resid = px.scatter(
        x=list(range(len(residuals))),
        y=residuals,
        labels={'x': 'Sample Index', 'y': 'Residual (Actual - Predicted)'}
    )
    fig_resid.update_traces(marker=dict(size=4, color='red'))
    fig_resid.update_layout(template="plotly_white", hovermode='x unified')
    st.plotly_chart(fig_resid, use_container_width=True)

# ---------- Feature Importance ----------
    if model_choice in ["Random Forest", "Decision Tree"]:
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
            st.subheader("📊 Feature Importance")
            fig_imp = px.bar(
                importances,
                x=importances.values,
                y=importances.index,
                orientation='h',
                labels={'x': 'Importance Score', 'index': 'Feature'},
                title=f"Feature Importance - {model_choice}"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------
# 🤖 Tab 3: Model Comparison Table
# -------------------------------
with tab3:
    st.divider()
    st.subheader("📊 Comparison of All Models")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Decision Tree": DecisionTreeRegressor(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
    }

    results = []

    for name, mdl in models.items():
        mdl.fit(xtrain, ytrain)
        preds_m = mdl.predict(xtest)
        r2_val = r2_score(ytest, preds_m)
        rmse_val = root_mean_squared_error(ytest, preds_m)
        mae_val = mean_absolute_error(ytest, preds_m)
        results.append({
            "Model": name,
            "R² Score": round(r2_val, 4),
            "RMSE": round(rmse_val, 4),
            "MAE": round(mae_val, 4)
        })

    # Add Deep Learning results (from previous DNN execution)
    results.append({
        "Model": "Deep Neural Network",
        "R² Score": round(r2_score(ytrue, preds), 4),
        "RMSE": round(root_mean_squared_error(ytrue, preds), 4),
        "MAE": round(mean_absolute_error(ytrue, preds), 4)
    })

    # Display comparison
    st.markdown("""
    <div style="background-color:#f9f9f9;padding:12px;border-radius:6px;border-left:5px solid #4A90E2;">
    <b>📌 Interpretation Guide:</b><br>
    🟦 <b>Blue:</b> Best <u>R² Score</u> — higher values are better.<br>
    🟩 <b>Green:</b> Best <u>RMSE</u> and <u>MAE</u> — lower values are better.<br>
    </div>
    """, unsafe_allow_html=True)
    results_df = pd.DataFrame(results).set_index("Model")
    st.dataframe(
        results_df.style.highlight_min(axis=0, color="lightgreen").highlight_max(axis=0, color="lightblue"),
        use_container_width=True
    )