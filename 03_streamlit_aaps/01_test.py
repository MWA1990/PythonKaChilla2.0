import streamlit as st
import seaborn as sns

st.header("This Video is brought to you by #BabaAmmar")
st.text('Kia ap ko maza araha hai')

df = sns.load_dataset('iris')
st.write(df[['species', 'sepal_length', 'petal_length']].head(10))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])