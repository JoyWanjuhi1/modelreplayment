import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time 

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🔮 Species Predictor Engine")
st.write("Adjust the sliders below to compute real-time classification probabilities.")
st.divider()

# Input sliders grouped beautifully
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

st.divider()

# Predict action
if st.button("Run Model Prediction", width='stretch'): # Updated to modern layout parameter
    
    with st.spinner("🧠 Computing model classification paths..."):
        time.sleep(0.7) 
        
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        flower_names = ["Setosa", "Versicolor", "Virginica"]
        predicted_flower = flower_names[prediction]
        confidence = probability[prediction] * 100
        
        # Display results with metrics layout
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.success(f"**Predicted Species:** {predicted_flower}")
        with res_col2:
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
            
        st.write("### 📈 Class Probabilities Breakdown")
        prob_df = pd.DataFrame({
            'Species': flower_names,
            'Probability (%)': [p * 100 for p in probability]
        }).set_index('Species')
        
        st.bar_chart(prob_df)
        
    st.toast(f"Successfully classified as {predicted_flower}!", icon="🎯")


st.divider()

# Bulk Processing (CSV Upload)
st.write("### 📂 Bulk Processing (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file containing iris measurements to predict in batch:", type=["csv"])

if uploaded_file is not None:
    bulk_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(bulk_data.head(), width='stretch') # Updated to modern layout parameter
    
    if st.button("Generate Bulk Predictions", width='stretch'): # Updated to modern layout parameter
        bulk_preds = model.predict(bulk_data)
        bulk_data['Predicted_Species'] = [["Setosa", "Versicolor", "Virginica"][p] for p in bulk_preds]
        
        st.success("🎉 Batch Processing Complete!")
        st.dataframe(bulk_data, width='stretch') # Updated to modern layout parameter
        
        csv_data = bulk_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Report as CSV",
            data=csv_data,
            file_name="iris_model_predictions.csv",
            mime="text/csv",
            width='stretch' # Updated to modern layout parameter
        )