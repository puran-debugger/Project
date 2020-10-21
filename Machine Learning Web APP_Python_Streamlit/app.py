import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# cd
# streamlit run app.py

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title('Binary Classification Web App')
    # mushroom emoji
    st.markdown('Are you mushrooms edible or poisonous?🍄')
    st.sidebar.markdown('Are you mushrooms edible or poisonous?🍄')
    
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushroom.csv')
        label = labelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    df = load_data()
    
    
    
if __name__ == '__main__':
    main()





