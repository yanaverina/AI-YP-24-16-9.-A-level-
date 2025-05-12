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


def fit_request(X_train: pd.DataFrame, y_train, model_id, model_type, hyperparams):
    #url = 'http://localhost:8000/models/fit/'
    url = 'http://fastapi-app:8000/api/v1/models/fit/'
    X = X_train.to_dict(orient='list')
    data = {
        'model_id': model_id,
        'model_type': model_type,
        'X': X,
        'y': y_train.tolist(),
        'hyperparams': hyperparams
    }
    response = requests.post(url, json=data)
    fit_data = response.json()

    for i in range(len(target_classes)):
        st.write(f'<p style="font-size: 20px;"><strong>Curves for "{target_classes[i]}" class</strong></p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            df = pd.DataFrame({
                'fpr': fit_data['data']['roc_curve'][str(i)]['fpr'],
                'tpr': fit_data['data']['roc_curve'][str(i)]['tpr']
            })
            fig = px.line(df, x="fpr", y="tpr", title=f"ROC Curve", template="plotly_white", width=600, height=600)
            st.plotly_chart(fig)

        with col2:
            df2 = pd.DataFrame({
                'recall': fit_data['data']['pr_curve'][str(i)]['recall'],
                'precision': fit_data['data']['pr_curve'][str(i)]['precision']
            })
            fig2 = px.line(df2, x="recall", y="precision", title=f"Precision Recall Curve", template="plotly_white", width=600, height=600)
            st.plotly_chart(fig2)

def set_models(model_id):
    #url = f'http://localhost:8000/models/set_model?model_id={model_id}'
    url = f'http://fastapi-app:8000/api/v1/models/set_model?model_id={model_id}'
    response = requests.post(url, json=model_id)
    result = response.json()
    st.write(result['message'])

def prediction(X_test: pd.DataFrame):
    #url = 'http://localhost:8000/models/predict/'
    url = 'http://fastapi-app:8000/api/v1/models/predict/'
    X_test_dict = X_test.to_dict(orient='list')
    data = {'X':X_test_dict}
    response = requests.post(url, json=data)
    fit_data = response.json()
    
    df = pd.DataFrame({
                'question': X_test_dict['question'],
                'prediction class': fit_data['data']['y_pred'],
            })

    return df

def models_list():
    #url = 'http://localhost:8000/models/list_models/'
    url = 'http://fastapi-app:8000/api/v1/models/list_models/'
    response = requests.get(url)
    result = response.json()
    st.write(result['message'])
    st.write(result['data'])



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
        
        model_type = st.selectbox('Выбор модели', ['log_reg'], key='model_type')
        if model_type is not None:
            if model_type == 'log_reg':
                hyperparams_C = st.selectbox('Выбор коэффициента C', [1, 10, 100], key='hyperparams_C')
                hyperparams_max_iter = st.selectbox('Выбор количества итераций', [100, 1000, 10000], key='hyperparams_max_iter')
                st.write('Обязательный выбор мультиклассовой классификации :)')
                hyperparams = {'C': hyperparams_C, 'max_iter': hyperparams_max_iter, 'multi_class' : 'ovr'}
                fit_request(X_train, y_train, model_id='log_reg_new', model_type=model_type, hyperparams=hyperparams)
    
    with tab_list:
        models_list()

    with tab_pred:
        model_type = st.selectbox('Выбор модели', ['log_reg', 'naive_bayes'])
        if model_type == 'log_reg':
            set_models('log_reg_baseline')
        else:
            set_models('naive_bayes_baseline')

        if model_type is not None:
            X_testing = st.file_uploader('Загрузите Датасет для определения его тематики',
                     type=allowed_extensions)

            if not X_testing:
                st.warning('Пожалуйста, загрузите датасет')
                return
        
            df_test = pd.read_csv(X_testing)
            df_test = df_test[df_test['target'].isin(target_classes)]
            df_test['qst_processed'] = df_test['question'].apply(preprocess_text)
            df_test['qst_len'] = df_test['qst_processed'].apply(len)
            df_result = prediction(df_test)

            y_thema = encoder.inverse_transform(df_result['prediction class'])
            df_result['predicted_thema'] = y_thema
            st.dataframe(df_result)





if __name__ == "__main__":
    main()

