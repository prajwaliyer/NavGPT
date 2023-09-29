from flask import Flask, render_template, request
from dataclasses import dataclass
import pandas as pd
import openai
import tiktoken
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import googleapiclient.discovery
import json
from google.oauth2 import service_account
import google.auth.transport.requests
import requests
import time

from functions import create_df, generate_embeddings, generate_query_embeddings, top_5_results

load_dotenv()
openai.organization = os.getenv("ORG_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

credentials = service_account.Credentials.from_service_account_file("youtube-api-credentials.json", scopes=["https://www.googleapis.com/auth/youtube.readonly","https://www.googleapis.com/auth/youtube.force-ssl"])
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY, credentials=credentials)

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query')
    data = create_df(query)
    data_embedding = generate_embeddings(data)
    query_embedding = generate_query_embeddings(query)
    results = top_5_results(data_embedding, query_embedding)
    time.sleep(2)
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run()



