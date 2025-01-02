import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import datetime as dt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any

st.set_page_config(page_title='Классификация вопросов экзамена A-level по темам', layout="wide")

nltk.download('punkt')
nltk.download('stopwords')
#nlp = spacy.load("en_core_web_sm")
nltk.download('punkt_tab')
nltk.download('wordnet')

expected_columns = {'file', 'page', 'question', 'score', 'target'}

target_classes = ['marketing mix and strategy', 'entrepreneurs and leaders',
       'the market', 'managing people', 'meeting customer needs']

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    wnl = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [wnl.lemmatize(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

def plot_classes_hist(df: pd.DataFrame):

    fig = px.histogram(df, x="target", 
                       color="target",
                       labels={
                           'target': 'Тематика вопроса',
                       })
    
    fig.update_layout(
        xaxis_title="Тематика вопроса",
        yaxis_title="Частота встречаемости тематики",
        title="Гистограмма распределения тематик вопросов"
    )

    st.plotly_chart(fig)


def fit_request(X_train: pd.DataFrame, y_train, model_id, model_type, hyperparams=None):
    url = 'http://fastapi-app:8000/api/v1/models/fit/'
    X = X_train.to_dict(orient='list')
    data = {
        'model_id': model_id,
        'model_type': model_type,
        'X': X,
        'y': y_train.tolist(),
    }
    response = requests.post(url, json=data)
    fit_data = response.json()



def plot_wordcloud(df: pd.DataFrame, target_class: str):
    df_target = df[df['target']==target_class]
    questions_joined = ' '.join(df_target['qst_processed'])
    wordcloud = WordCloud(width=800, 
                          height=400, 
                          background_color='white'
                        ).generate(questions_joined)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Облако слов - {target_class}', fontsize=16)
    st.pyplot(fig)


def infer_with_trained_model(model, X_infer):
    predictions = model.predict(X_infer)
    st.write(predictions)
    probabilities = model.predict_proba(X_infer)
    st.write(probabilities)
    return predictions, probabilities


def main():
    st.title("Классификация вопросов экзамена A-level по темам")
    allowed_extensions = ['.csv']
    data_file = st.file_uploader('Загрузите Датасет',
                     type=allowed_extensions)
    
    if not data_file:
        st.warning('Пожалуйста, загрузите датасет')
        return
    
    df = pd.read_csv(data_file)
    
    if set(df.columns) != expected_columns:
        st.warning(f'''Пожалуйста, загрузите датасет с правильными столбцами:
                   {', '.join(expected_columns)}
                   ''')
        return
    df = df[df['target'].isin(target_classes)]
    df['qst_processed'] = df['question'].apply(preprocess_text)
    df['qst_len'] = df['qst_processed'].apply(len)
    

    tab_eda, tab_fit, tab_pred, tab_list = st.tabs(['EDA',
                                                    'Обучить модель',
                                                    'Предсказание на test',
                                                    'Список моделей'])
    
    with tab_eda:
        plot_classes_hist(df)
        
        target_class = st.selectbox('Выберите тематику', 
                                       options=target_classes,
                                       key='themes_selector')
        
        st.subheader(f'Аналитика для вопросов с тематикой {target_class}')

        plot_wordcloud(df, target_class)

    with tab_fit:
        X = df.copy()
        
        encoder = LabelEncoder()
        y = encoder.fit_transform(df['target'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_type = st.selectbox('Выбор модели', ['naive_bayes', 'log_reg'])

        fit_request(X_train, y_train, model_id='log_reg_new', model_type=model_type)



if __name__ == "__main__":
    main()