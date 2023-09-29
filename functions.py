from dataclasses import dataclass
import pandas as pd
import openai
import tiktoken
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
from google import auth
from youtube_transcript_api import YouTubeTranscriptApi
import googleapiclient.discovery
import json
from google.oauth2 import service_account
import google.auth.transport.requests
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import concurrent.futures
import time

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
        except:
              transcript.append([{'text': 'None' , 'start':0.00,'duration':0.00}])
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
       tn = item["snippet"]["thumbnails"]["high"]["url"]
       data['Link'].append(video_link)
       data['Title'].append(title)
       data['Channel_title'].append(channel_title)
       data['Thumbnail'].append(tn)
    
    df_meta = pd.DataFrame(data)
    df = pd.concat([df_meta,df_ft,df_parts],axis = 1) 
    df = df.dropna(subset = 'Full Text') 
    df = df[df['Full Text'] != 'None ']

    # df.to_excel("data/youtube_transcripts.xlsx")
    print("Dataframe created")
    return df

def generate_single_embedding(text):
    # Generate embedding for a single text
    return get_embedding(text, engine=embedding_model)

def generate_embeddings(df):
    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Iterate through the rows of the dataframe
        for index, row in df.iterrows():
            # Prepare a list to hold the text segments for batch processing
            texts = []
            # Iterate through the columns containing text segments
            for col in row.index[5:]:
                if not pd.isna(row[col]):
                    # Remove newlines and sound indicators
                    row[col]['text'] = row[col]['text'].replace("\n", " ")
                    row[col]['text'] = re.sub(r"\[.*?\]", "", row[col]['text'])
                    # Combine title of video and text
                    combined = f"Title: {row['Title']}; Channel: {row['Channel_title']}; Content: {row[col]['text']}"
                    # Add the combined text to the list
                    texts.append(combined)
                else:
                    break
            # Generate embeddings for all text segments in the batch
            embeddings = list(executor.map(generate_single_embedding, texts))
            # Add the embeddings to the dataframe
            for i, col in enumerate(row.index[5:]):
                if i < len(embeddings):
                    row[col]['embedding'] = embeddings[i]

    print("Data embeddings generated")
    return df

def generate_query_embeddings(query):

    embedding = get_embedding(query, engine=embedding_model)
    print("Query embeddings generated")
    return embedding

def top_5_results(data_embedding, query_embedding):
    
    similarities = []
    
    # Iterate through the rows of the dataframe
    for index, row in data_embedding.iterrows():
        # Iterate through the columns containing text segments
        for col in row.index[5:]:
            if not pd.isna(row[col]):
                # Compute cosine similarity
                similarity = cosine_similarity([row[col]['embedding']], [query_embedding])[0][0]
                similarities.append((similarity, row[col]['text'], row[col]['start'], row['Title'], row['Channel_title'], row['Link'], row['Thumbnail']))
                
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Get the top 5 transcripts
    top_transcripts = [row[1] for row in similarities[:5]]

    # Generate summaries for the top 5 transcripts
    summaries = generate_summaries(top_transcripts)

    updated_similarities = []
    for i in range(min(5, len(similarities))):
        row = similarities[i]
        summary = summaries[i]
        updated_row = row + (summary,)
        updated_similarities.append(updated_row)
    
    print("Similarities ranked")
    return updated_similarities[:5]

def generate_single_summary(transcript):
    time.sleep(0.2)  # add a delay of 200 milliseconds
    prompt = f"Create a one-line summary of the following transcript:\n\n{transcript}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who summarizes paragraphs in one sentence."},
            {"role": "user", "content": prompt}
        ]
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def generate_summaries(transcripts):
    # Create a ThreadPoolExecutor with a maximum of 5 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Generate summaries for all transcripts in the batch
        summaries = list(executor.map(generate_single_summary, transcripts))

    print("Summaries generated")
    return summaries