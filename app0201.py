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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import datetime as dt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title='Классификация вопросов экзамена A-level по темам', layout="wide")

nltk.download('punkt')
nltk.download('stopwords')
#nlp = spacy.load("en_core_web_sm")

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

def create_and_train_model(X_train, y_train, model_type, hyperparams=None):
    if model_type == 'MultinomialNB':
        clf = MultinomialNB(**hyperparams)
    elif model_type == 'LogisticRegression':
        clf = LogisticRegression(**hyperparams)
    else:
        st.error('Неправильный тип модели')
        return None
    
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf),
    ])
    
    text_clf.fit(X_train, y_train)
    return text_clf

def view_model_info(model, y_test, X_test):
    if isinstance(model, Pipeline):
        st.write('Модель состоит из следующих шагов:')
        for step_name, step in model.steps:
            st.write(f'- {step_name}: {type(step).__name__}')
            
            if hasattr(step, 'get_params'):
                params = step.get_params()
                st.json(params)
                
    st.write('Метрики классификации:')
    test_pred = model.predict(X_test)
    report = classification_report(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    st.write(report)
    st.write('Матрица путаницы:')
    st.dataframe(pd.DataFrame(cm))

def infer_with_trained_model(model, X_infer):
    predictions = model.predict(X_infer)
    st.write(predictions)
    probabilities = model.predict_proba(X_infer)
    st.write(probabilities)
    return predictions, probabilities


def main():
    st.title("Классификация вопросов экзамена A-level по темам")
    allowed_extensions = ['.csv']
    data_file = st.file_uploader('Загрузите файл с историческими данными о погоде',
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
    

    tab_eda, tab_models = st.tabs(['EDA', 'Модели'])
    
    with tab_eda:
        plot_classes_hist(df)
        
        target_class = st.selectbox('Выберите тематику', 
                                       options=target_classes,
                                       key='themes_selector')
        
        st.subheader(f'Аналитика для вопросов с тематикой {target_class}')

        plot_wordcloud(df, target_class)

    with tab_models:
        X = df['qst_processed']
        
        encoder = LabelEncoder()
        y = encoder.fit_transform(df['target'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_type = st.selectbox('Выбор модели', ['LogisticRegression', 'MultinomialNB'])
        if model_type == 'LogisticRegression':
            hyperparams = {'multi_class': 'ovr'}
        else:
            hyperparams = {}
        
        if st.button('Обучить модель'):
            trained_model = create_and_train_model(X_train, y_train, model_type, hyperparams)
            if trained_model:
                st.success('Модель успешно обучена!')
                dump(trained_model, f'{model_type}_model.joblib')
                st.balloons()
        
        if st.button('Посмотреть информацию о модели'):
            loaded_model = load(f'{model_type}_model.joblib')
            view_model_info(loaded_model, y_test, X_test)
        
        if st.button('Сделать прогноз'):
            question = st.text_input('Введите вопрос для классификации')
            if question:
                prediction, probabilities = infer_with_trained_model(loaded_model, [question])
                st.write(prediction)
                st.write(probabilities)
                st.write(f'Предсказание: {prediction[0]}')
                st.write('Вероятности классов:')
                st.dataframe(pd.DataFrame(probabilities, columns=loaded_model.classes_))

if __name__ == "__main__":
    main()