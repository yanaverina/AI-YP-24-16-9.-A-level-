FROM python:3.12

WORKDIR /app-streamlit

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install nltk
RUN mkdir -p /usr/share/nltk_data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt stopwords
RUN ls /usr/share/nltk_data/tokenizers/
ENV NLTK_DATA=/usr/share/nltk_data

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
