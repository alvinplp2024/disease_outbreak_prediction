import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model  # type: ignore

# Load model components
model = load_model('disease_prediction_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Retrieve the expected features (the names the scaler was fit on)
expected_features = list(scaler.feature_names_in_)

st.set_page_config(page_title="Disease Outbreak Predictor", layout="wide")
st.title("🦠 Predict Next Disease Outbreak")

uploaded_file = st.file_uploader("📤 Upload CSV file with environmental features", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅File uploaded successfully.")

        # Report which expected features are available; fill missing ones with 0.
        available_features = list(set(expected_features) & set(df.columns))
        st.info(f"Found {len(available_features)} features in your file that match expected features: {available_features}")
        
        # Build input DataFrame with exactly the expected features.
        # For each expected feature, if it exists, convert to numeric; otherwise, fill with 0.
        X_input = pd.DataFrame(columns=expected_features)
        for feat in expected_features:
            if feat in df.columns:
                X_input[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
            else:
                X_input[feat] = 0

        # Reorder the columns to match the expected feature order.
        X_input = X_input[expected_features]
        
        # Debug: Show the first few rows of the preprocessed features.
        st.write("Preprocessed features (first 5 rows):", X_input.head())

        # Scale the input features.
        X_scaled = scaler.transform(X_input)

        # Reshape for LSTM: (samples, timesteps, features).
        X_reshaped = np.expand_dims(X_scaled, axis=1)

        # Make predictions.
        preds = model.predict(X_reshaped)
        
        # For each row, extract indices of the top 3 predictions (ordered highest first).
        top_3 = np.argsort(preds, axis=1)[:, -3:][:, ::-1]

        # Decode the top 3 predictions into class labels.
        top_3_labels = np.vectorize(lambda x: label_encoder.inverse_transform([x])[0])(top_3)

        # Create a DataFrame to display the per-row results in a table.
        results_df = pd.DataFrame(top_3_labels, columns=['Top 1', 'Top 2', 'Top 3'])
        st.subheader("🔮 Predicted Top 3 Likely Outbreaks (per sample)")
        st.dataframe(results_df)

        # Download button for saving the table.
        st.download_button(
            "📥Download predictions",
            results_df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

        # EXTRA SECTION:
        # Aggregate the top-1 predicted disease from each row and find the overall top three.
        top1_labels = results_df["Top 1"]
        overall_counts = top1_labels.value_counts()
        top_overall = overall_counts.head(3).index.tolist()

        # Display the accumulated top predicted diseases as a list.
        st.subheader("🔮Overall Top Predicted Diseases")
        for idx, disease in enumerate(top_overall, start=1):
            st.write(f"{idx}. {disease}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
