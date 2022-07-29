# importing Libraries
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Aap ki Heading
st.write('''
# Explore Different ML Models and Datasets
Let see which one is best.
''')

# Dataset k naam ek sidebar box main
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# Classifier k naam ek sidebar box main
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM','Random Forest')
)

# Define a function to load dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y

# Ab is function ko bula lay gay or x,y ko equal rakh lain
X, y = get_dataset(dataset_name)

# Ab hum apny DataSet ki shape ko print karain gey
st.write('Shape of DataSet:', X.shape)
st.write('Number of Classes:', len(np.unique(y)))

# Next Different Classifiers k parameters ko user input main add karain gey
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('Max Depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

# Ab hum Classifier bnain gey based on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C = params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators= params['n_estimators'], max_depth= params['max_depth'], random_state=1234)
    return clf

# To Show source code with check box
if st.checkbox('Show Code'):
    with st.echo():
        clf = get_classifier(classifier_name, params)

        # Train and Test Split into 80 / 20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        # Ab hum classifier ko train karain gey
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Model Accuracy check and print
        acc = accuracy_score(y_test, y_pred)
        
clf = get_classifier(classifier_name, params)

# Train and Test Split into 80 / 20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Ab hum classifier ko train karain gey
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Model Accuracy check and print
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = ', acc)

# Plot the data
pca = PCA(2)
X_projected = pca.fit_transform(X)

# Slice data into 0 or 1 dimension
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)
