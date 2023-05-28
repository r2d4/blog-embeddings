import json
import os
import time

import numpy as np
import openai
import pandas as pd
import torch
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer


def read_json_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def generate_category(titles):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    titles_str = ", ".join(titles)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""These are the titles: {titles_str}. What is the overarching topic of these titles? 
- Be as specific as possible. 
- Only use a few words. 
- You are writing for a highly technical audience. 
- Only respond with the topic and nothing else."""},
    ]

    while True:
        try:
            # Use the chat models
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                timeout=5,  
            )

            category = response['choices'][0]['message']['content']
            print(f"Category: {category}")
            return category
        except openai.error.OpenAIError:
            print("An error occurred. Retrying...")
            time.sleep(2)


def embedding_metadata(embeddings_2d, titles, grid_size=20):
    df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df["title"] = titles

    x_min, x_max = (df["x"].min() // grid_size) * grid_size, (df["x"].max() // grid_size + 1) * grid_size
    y_min, y_max = (df["y"].min() // grid_size) * grid_size, (df["y"].max() // grid_size + 1) * grid_size

    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    df["x_bin"] = np.digitize(df["x"], x_bins)
    df["y_bin"] = np.digitize(df["y"], y_bins)

    groups = df.groupby(["x_bin", "y_bin"])["title"].apply(list)

    ret = []
    for (x_bin, y_bin), titles in groups.items():
        x_bounds = (x_bins[x_bin-1], x_bins[x_bin-1] + grid_size)
        y_bounds = (y_bins[y_bin-1], y_bins[y_bin-1] + grid_size)
        category = generate_category(titles)

        ret.append({
            "x_bounds": x_bounds,
            "y_bounds": y_bounds,
            "titles": titles,
            "category": category
        })

    return ret

def count_elements(embeddings_2d):
    x_values, y_values = embeddings_2d[:,0], embeddings_2d[:,1]

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    quadrants = {
        "Q1": {"x": (x_mid, x_max), "y": (y_mid, y_max)},
        "Q2": {"x": (x_min, x_mid), "y": (y_mid, y_max)},
        "Q3": {"x": (x_min, x_mid), "y": (y_min, y_mid)},
        "Q4": {"x": (x_mid, x_max), "y": (y_min, y_mid)}
    }

    counts = {q: 0 for q in quadrants.keys()}

    for x, y in embeddings_2d:
        for q, bounds in quadrants.items():
            if bounds["x"][0] <= x <= bounds["x"][1] and bounds["y"][0] <= y <= bounds["y"][1]:
                counts[q] += 1

    return counts

def filter_posts(posts):
    filtered_posts = [post for post in posts if post['status'] == 'published' and post['plaintext'] is not None and post['title'] is not None]
    return filtered_posts

def extract_titles_texts(filtered_posts):
    titles = [post['title'] for post in filtered_posts]
    texts = [post['plaintext'] for post in filtered_posts]
    return titles, texts

def get_data():
    data = read_json_data('data/blog.json')
    posts = data['db'][0]['data']['posts']
    filtered_posts = filter_posts(posts)
    titles, texts = extract_titles_texts(filtered_posts)
    return titles, texts

def load_model_and_tokenizer(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def get_embeddings(tokenizer, model, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def reduce_dimensions(embeddings, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

def write_embeddings_to_json(embeddings_2d, titles, json_file_path):
    data = []
    for i, title in enumerate(titles):
        data.append({
            'title': title,
            'x': embeddings_2d[i, 0].tolist(), 
            'y': embeddings_2d[i, 1].tolist()
        })

    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)