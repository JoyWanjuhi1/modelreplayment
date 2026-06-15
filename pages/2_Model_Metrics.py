import streamlit as st
import pandas as pd

st.title("📊 Model Evaluation Metrics")
st.write("Below are the performance metrics for the underlying classification model.")
st.divider()

# Display core model KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Overall Accuracy", "96.7%")
col2.metric("Precision (Macro)", "0.97")
col3.metric("Recall (Macro)", "0.96")

st.write("### 📝 Classification Report")
# Mock report matching your performance metrics
report_data = {
    "Species": ["Setosa", "Versicolor", "Virginica"],
    "Precision": [1.00, 0.93, 0.96],
    "Recall": [1.00, 0.96, 0.93],
    "F1-Score": [1.00, 0.95, 0.95]
}
st.table(pd.DataFrame(report_data).set_index("Species"))

st.info("💡 **Analyst Note:** The model struggles slightly distinguishing between Versicolor and Virginica due to overlapping petal dimensions, while Setosa remains perfectly linearly separable.")