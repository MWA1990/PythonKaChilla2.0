import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# make containers
header = st.container()
data_sets = st.container()
featuress = st.container()
model_training = st.container()

with header:
    st.title('Kashti ki Aap')
    st.text('In the project we will work on kashti data')

with data_sets:
    st.header('Kashti dhoob gae')
    st.text('we will work on titanic dataset')
    # import dataset
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))
    st.subheader('Kitney admay thy...')
    st.bar_chart(df['sex'].value_counts())

    # other plots
    st.subheader('Class k hisab say farq')
    st.bar_chart(df['class'].value_counts())

    # barplot
    st.bar_chart(df['age'].sample(10))

with featuress:
    st.header('These are our features')
    st.text('awen bht saray features aad karny hain')
    st.markdown('1. **Feature 1:** This will tell us pata nahe')

with model_training:
    st.header('Kashti walon ka kiya bana? model training')
    st.text('kuch karna hai')
    # making columns
    input, display = st.columns(2)

    # pehlay column main ap ke selection points hain
    max_depth = input.slider('How many people do you know?', min_value=10, max_value= 100, value = 20, step=5)

# n_estimators
n_estimators = input.selectbox('How many tree should be there in a RF?', options=[50, 100, 200, 300, 'No Limit'])

# adding list of features
input.write(df.columns)

# input features from user
input_features = input.text_input('Which feature we should use?')

# machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# yahan per hum ek condition lagain gey
if n_estimators == 'No Limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)



# define x and y
x = df[[input_features]]
y = df[['fare']]

# fit model
model.fit(x,y)
pred = model.predict(y)

# display metrices
display.subheader('Mean Absolute error: ')
display.write(mean_absolute_error(y, pred))

display.subheader('Mean Squared error: ')
display.write(mean_squared_error(y, pred))

display.subheader('R^2 Squared Score: ')
display.write(r2_score(y, pred))