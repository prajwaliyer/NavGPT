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

from functions import combine_text_by_duration, create_df, generate_embeddings, generate_query_embeddings, top_3_results

YOUTUBE_API_KEY = "AIzaSyDCtUOqvJB9cEhsHPGKUtQSdjQg4zq8oC8"
ACCESS_TOKEN = "428637118276-8n95ohv3clke0hj3bdd4k5b7hgs72qr6.apps.googleusercontent.com"

credentials = service_account.Credentials.from_service_account_file("youtubecaptions-384401-fe74ae73c098.json", scopes=["https://www.googleapis.com/auth/youtube.readonly","https://www.googleapis.com/auth/youtube.force-ssl"])

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY, credentials=credentials)
session = google.auth.transport.requests.AuthorizedSession(credentials)

session.headers.update({'Authorization': 'Bearer ' + ACCESS_TOKEN})

load_dotenv()
openai.organization = os.getenv("ORG_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    results = top_3_results(data_embedding, query_embedding)
    print(data_embedding.head(2))
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run()



