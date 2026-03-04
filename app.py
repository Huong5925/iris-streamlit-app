import streamlit as st
import pickle
from sklearn.datasets import load_iris

iris = load_iris()

model = pickle.load(open("iris_model.pkl", "rb"))

st.title("Iris Flower Classification App")

st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

if st.sidebar.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"Prediction: {iris.target_names[prediction[0]]}")
