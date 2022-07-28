# import libraries
from doctest import Example
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport, profile_report
from streamlit_pandas_profiling import st_profile_report

# Webapp ka title
st.markdown(''' 
# **Exploratory Data Analysis Web Application**
This Aap is developed by Muhammad Waleed Anjum.
''')
st.markdown('Explore Titanic Example Dataset')

# How to upload a file from pc
with st.sidebar.header('Upload your Dataset (.csv)'):
    uploaded_file = st.sidebar.file_uploader('Upload your file', type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](df)")

# profiling report for pandas
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Profiling Report With Pandas**')
    st_profile_report(pr)
else:
    st.info('Awaiting for csv file')
    if st.button('press to use example data'):
    # example dataset
        @st.cache
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                                columns=['age', 'banana', 'codenics', 'dog', 'ear'])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)