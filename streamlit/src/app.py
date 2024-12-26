import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
i

st.set_page_config(
    page_title="Классификация вопросов экзамена A-level по темам",
    layout="wide"
)


expected_columns = {'file', 'page', 'question', 'score', 'target'}

target_classes = ['marketing mix and strategy', 'entrepreneurs and leaders',
       'the market', 'managing people', 'meeting customer needs']

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    doc = nlp(' '.join(filtered_tokens))
    lemmatized_tokens = ' '.join([token.lemma_ for token in doc])
    return lemmatized_tokens

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

    # Отбираем только те классы, которые может прогнозировать наша модель
    df = df[df['target'].isin(target_classes)]
    
    tab_eda, tab_models = st.tabs(['EDA', 'Модели'])
    
    with tab_eda:
        plot_classes_hist(df)


    

        

    


if __name__=='__main__':
    main()