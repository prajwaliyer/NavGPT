from flask import Flask, render_template, request
from dataclasses import dataclass
import pandas as pd
import openai
import tiktoken
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
from google.colab import auth
auth.authenticate_user()
from youtube_transcript_api import YouTubeTranscriptApi
import googleapiclient.discovery
import json
from google.oauth2 import service_account
import google.auth.transport.requests
import pandas as pd
import requests

#print(requests.get('https://api.ipify.org').text)

API_KEY = "AIzaSyDCtUOqvJB9cEhsHPGKUtQSdjQg4zq8oC8"
ACCESS_TOKEN = "428637118276-8n95ohv3clke0hj3bdd4k5b7hgs72qr6.apps.googleusercontent.com"

credentials = service_account.Credentials.from_service_account_file("/content/youtubecaptions-384401-fe74ae73c098.json", scopes=["https://www.googleapis.com/auth/youtube.readonly","https://www.googleapis.com/auth/youtube.force-ssl"])

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY, credentials=credentials)
session = google.auth.transport.requests.AuthorizedSession(credentials)

session.headers.update({'Authorization': 'Bearer ' + ACCESS_TOKEN})

def combine_text_by_duration(data, segment_duration):
    combined_text = []
    current_text = ""
    current_duration = 0
    for item in data:
  
        item_duration = item['duration']
        item_text = item['text']
        
        while item_duration > 0:
            dcd = segment_duration - current_duration
            if dcd == segment_duration:
                item_start = item['start']

            time_to_add = min(dcd, item_duration)
            current_text += item_text + ' '
            item_duration -= time_to_add
            current_duration += time_to_add
            
            if current_duration == segment_duration:
                combined_text.append({'text': current_text, 'start': item_start})
                fulltext.append(current_text)
                current_text = ""
                current_duration = 0
        
    return combined_text

def create_df(query):
    
    search_response = youtube.search().list(
    q=query,
    type="video",
    part="id,snippet",
    fields="items(id,snippet)",
    maxResults=10,
    ).execute()
    transcript=[]
    thumbnail=[]
    for video_result in search_response["items"]:
        video_id = video_result["id"]["videoId"]
        try:
            transcript.append(YouTubeTranscriptApi.get_transcript(video_id))
            #print(transcript)
        except:
              continue
    fulltext = []

    for i in range(len(transcript)):
        combined_text = ''
        for item in transcript[i]:
            combined_text += item['text'] + ' '
        fulltext.append([combined_text])
    df_ft = pd.DataFrame(fulltext)
    df_ft.columns=['Full Text']
    df_ft.head()     
    y=[]
    for i in range(len(transcript)):
        y.append(combine_text_by_duration(transcript[i],180))
    df_parts = pd.DataFrame(y)
    data = {'Channel_title': [], 'Title': [], 'Link': [], 'Thumbnail': []}
    for item in search_response["items"]:
       title = item['snippet']['title']
       channel_title = item['snippet']['channelTitle']
       video_link = "https://www.youtube.com/watch?v=" + item["id"]["videoId"]
       tn = video_result["snippet"]["thumbnails"]["high"]["url"]
       data['Link'].append(video_link)
       data['Title'].append(title)
       data['Channel_title'].append(channel_title)
       data['Thumbnail'].append(tn)
   df_meta = pd.DataFrame(data)
   df = pd.concat([df_meta,df_ft,df_parts],axis = 1) 
   df = df.dropna(subset = 'Full Text') 
    
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



