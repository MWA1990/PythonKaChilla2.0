# import libraries
from turtle import width
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


# import dataset
st.title ('Plotly and Streamlit ko mila k app bnana')
df = px.data.gapminder()
st.write(df)
#st.write(df.head())
st.write(df.columns)

# Summary Statistics
st.write(df.describe())

# Data Management  
year_option = df['year' ].unique().tolist()
year = st.selectbox('Which year should we plot?', year_option, 0)
#df = df[df['year'] == year]

# Plotting
fig = px.scatter(df, x='gdpPercap', y='lifeExp', size='pop', color='continent', hover_name='continent', 
                log_x=True, size_max=55, range_x=[100, 100000], range_y=[20, 90],
                animation_frame='year', animation_group='country')
fig.update_layout(width=600, height=600)
st.write(fig)