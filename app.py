from flask import Flask, render_template, request
from dataclasses import dataclass
import pandas as pd
import openai
import tiktoken
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

load_dotenv()
openai.organization = os.getenv("ORG_KEY")
openai.api_key = os.getenv("API_KEY")

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

app = Flask(__name__)

def create_embeddings():

    # Gather data
    df = pd.read_csv('data/YouTube_transcripts_Kaggle.csv')
    data = df[['title', 'author', 'transcript', 'playlist_name']]

    # Remove sound indicators
    data['transcript'] = data['transcript'].str.replace(r"\[.*?\]","", regex=True)
    del df

    # Combine title and transcripts
    data['combined'] = (
        "Title: " + data.title.str.strip() + "; Content: " + data.transcript.str.strip()
    )
    print(data.head(2))

    # Tokenize
    top_n = 5000
    encoding = tiktoken.get_encoding(embedding_encoding)
    # Omit transcripts that are too long to embed since ada model has max_tokens size
    data['n_tokens'] = data.combined.apply(lambda x: len(encoding.encode(x)))
    data = data[data.n_tokens <= max_tokens].tail(top_n)
    print(len(data))
    

    # This may take a few minutes
    data['embedding'] = data.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
    data.to_csv("data/embeddings.csv")

@app.route('/')
def index():
    # create_embeddings()   # Used one time to generate the embeddings
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query')
    return f'Search results for: {query}'

if __name__ == '__main__':
    app.run()



