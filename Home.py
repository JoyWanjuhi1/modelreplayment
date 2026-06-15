# Create the Homepage & Data Visualization
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# 1. Page Configuration (Must always be the very first Streamlit command)
st.set_page_config(
    page_title="Iris Analytics Hub",
    page_icon="🌸",
    layout="wide"
)

# 2. App Title and Introduction
st.title("🌸 Iris Flower Data Analytics Hub")
st.write("""
Welcome! This interactive dashboard explores the famous Iris dataset and uses a Machine Learning model 
to predict flower species based on physical measurements. Use the sidebar to navigate.
""")

st.divider()

# Load dataset for EDA
@st.cache_data
def get_eda_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = [iris.target_names[i] for i in iris.target]
    return df

df = get_eda_data()

# 🛠️ WIDGET: The Toggle Switch (Now safely placed AFTER 'st' is initialized)
show_raw_data = st.toggle("🔍 Show Complete Raw Dataset", value=False)

if show_raw_data:
    st.subheader("📋 Raw Iris Dataset")
    st.dataframe(df, width='stretch')
else:
    st.info("💡 Tip: Flip the switch above to view the full 150-row dataset matrix.")

st.divider()

# Layout: Dataset view and interactive chart side-by-side
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), width='stretch')
    st.markdown(f"**Total Records:** {df.shape[0]} | **Features:** {df.shape[1] - 1}")

with col2:
    st.subheader("📊 Interactive Feature Explorer")
    
    # Dynamic widgets to let user choose axes
    features = [col for col in df.columns if col != 'Species']
    x_axis = st.selectbox("Select X-Axis Feature:", features, index=0)
    y_axis = st.selectbox("Select Y-Axis Feature:", features, index=2)
    
    # 🎨 WIDGET: The Color Picker (Injected right into the chart control layout)
    theme_color = st.color_picker("🎨 Customize Primary Chart Plot Color:", "#636EFA")
    
    # Plotly Scatter Plot using the dynamic theme color selection
    fig = px.scatter(
        df, x=x_axis, y=y_axis, color="Species",
        title=f"{x_axis} vs {y_axis}",
        color_discrete_sequence=[theme_color, "#EF553B", "#00CC96"]
    )
    st.plotly_chart(fig, width='stretch')