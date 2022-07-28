# import libraries
from turtle import width
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns


# import dataset
st.title ('Iris Data Set Aap by using Plotly and Streamlit')
df = sns.load_dataset('iris')
#st.write(df)
st.write(df.head())
st.title('Columns')
st.write(df.columns)

# Summary Statistics
st.title('Data Summary')
st.write(df.describe())

# Plotting
st.title('Plotting using Plotly')
fig = px.scatter(df, x='sepal_length', y='sepal_width', size='sepal_width', color='species', hover_name='species', log_x=True, log_y=True)
fig.update_layout(width=600, height=600)
st.write(fig)