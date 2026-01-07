# AI-Powered Fraud Detection Dashboard for Auditing
# Created by Cassim Julius
# Streamlit version: Interactive app with file upload and dynamic visuals.
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from io import BytesIO

st.title("AI-Powered Fraud Detection Dashboard")
st.markdown("Created by Cassim Julius. Upload transaction data (CSV) or use synthetic. AI detects anomalies like fraud in audits.")

# Sidebar for settings
st.sidebar.header("Settings")
use_synthetic = st.sidebar.checkbox("Use Synthetic Data (Demo)", value=True)
uploaded_file = st.sidebar.file_uploader("Upload CSV (Features only, no labels)", type="csv")
num_epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
threshold_percentile = st.sidebar.slider("Threshold Percentile", 90, 99, 95)

if use_synthetic or uploaded_file:
    if use_synthetic:
        # Synthetic Data (as original)
        num_normal = 9000
        num_fraud = 1000
        num_features = 10
        normal_data = np.random.randn(num_normal, num_features)
        fraud_shift = np.random.uniform(2, 5, num_features)
        fraud_data = np.random.randn(num_fraud, num_features) * 1.5 + fraud_shift
        features = np.vstack((normal_data, fraud_data))
        labels = np.hstack((np.zeros(num_normal), np.ones(num_fraud)))
        idx = np.random.permutation(len(features))
        features = features[idx]
        labels = labels[idx]
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        features = (features - mean) / std
        normal_idx = np.where(labels == 0)[0]
        fraud_idx = np.where(labels == 1)[0]
        train_size = int(0.8 * len(normal_idx))
        X_train = features[normal_idx[:train_size]]
        X_test_normal = features[normal_idx[train_size:]]
        X_test_fraud = features[fraud_idx]
        X_test = np.vstack((X_test_normal, X_test_fraud))
        y_test = np.hstack((np.zeros(len(X_test_normal)), np.ones(len(X_test_fraud))))
    else:
        # Load uploaded CSV (assume numeric features, no header/labels)
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        features = df.values.astype(float)
        # Assume unsupervised: Train on all, test on all (for real audits)
        X_train = features
        X_test = features
        y_test = np.zeros(len(features))  # Placeholder, no true labels

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Autoencoder Model (same as original)
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    if st.button("Train Model & Detect Fraud"):
        with st.spinner("Training AI model..."):
            losses = []
            for epoch in range(num_epochs):
                output = autoencoder(X_train)
                loss = criterion(output, X_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Threshold
        with torch.no_grad():
            train_recon = autoencoder(X_train)
            train_error = torch.mean((train_recon - X_train) ** 2, dim=1)
            threshold = np.percentile(train_error.numpy(), threshold_percentile)

        # Evaluation
        with torch.no_grad():
            test_recon = autoencoder(X_test)
            reconstruction_error = torch.mean((test_recon - X_test) ** 2, dim=1)
        y_pred = (reconstruction_error > threshold).numpy().astype(int)
        y_test_np = y_test.numpy()

        tp = np.sum((y_pred == 1) & (y_test_np == 1))
        fp = np.sum((y_pred == 1) & (y_test_np == 0))
        tn = np.sum((y_pred == 0) & (y_test_np == 0))
        fn = np.sum((y_pred == 0) & (y_test_np == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_test_np)

        st.success("Training Complete!")
        st.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
        st.write(f"Threshold: {threshold:.4f} | Detected Fraud: {np.sum(y_pred)} out of {len(y_pred)} transactions")

        # Visuals
        col1, col2, col3 = st.columns(3)
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(range(1, num_epochs + 1), losses)
            ax1.set_title('Training Loss')
            buf1 = BytesIO()
            fig1.savefig(buf1, format="png")
            st.image(buf1)
        with col2:
            errors_normal = reconstruction_error[y_test == 0].numpy()
            errors_fraud = reconstruction_error[y_test == 1].numpy()
            fig2, ax2 = plt.subplots()
            ax2.hist(errors_normal, bins=50, alpha=0.7, label='Normal', color='blue')
            ax2.hist(errors_fraud, bins=50, alpha=0.7, label='Fraud', color='red')
            ax2.axvline(threshold, color='black', linestyle='--')
            ax2.set_title('Error Distribution')
            ax2.legend()
            buf2 = BytesIO()
            fig2.savefig(buf2, format="png")
            st.image(buf2)
        with col3:
            labels_pie = ['Normal', 'Fraud']
            sizes = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
            fig3, ax3 = plt.subplots()
            ax3.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=['blue', 'red'])
            ax3.set_title('Detected Types')
            buf3 = BytesIO()
            fig3.savefig(buf3, format="png")
            st.image(buf3)
else:
    st.info("Select synthetic data or upload a CSV to start.")