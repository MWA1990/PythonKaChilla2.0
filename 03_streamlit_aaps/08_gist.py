import streamlit as st
from streamlit_embedcode import github_gist

link = 'https://gist.github.com/MWA1990/4c3818db8f1b502cceeed198c44db608'

st.write('Embed Github Gist:')

github_gist(link)