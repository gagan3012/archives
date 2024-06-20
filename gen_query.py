import json
import random
import huggingface_hub

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    return parser.parse_args()

client = OpenAI(api_key="sk-proj-")

def respond_gpt4(
    message,
    system_message="""
    You are a smart and helpful assistant.
    Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following keys:
"query": a string, a random user search query specified by the retrieval task.
"positive": This will be provided to you. It is a list of strings, each representing a positive example of a document that should be retrieved by the search query.

Please adhere to the following guidelines:
- The "query" should be a random user search query.
- The "query" should be paragraph-based, in at least 10 words, understandable with some effort or ambiguity, and diverse in topic.
- Query should be strongly related to the "positive" examples given to you.
- Your input will be just the text which is to be considered as the positive examples.

""",
    max_tokens=512,
    temperature=0.2,
    top_p=0.95,
):
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": message})

    print(message)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages = messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content


def split_json(text):
    json_text = json.loads(text)
    return json_text['query'], json_text['positive']

def generate_query(file, output_file):
    df = pd.read_json(file, lines=True)
    df['gen'] = [respond_gpt4(df['text'][i]) for i in tqdm(range(len(df)))]

    df['query'], df['positive'] = zip(*df['gen'].map(split_json))

    df.drop(columns=['gen'], inplace=True)
    df.to_json(file, orient='records', lines=True)

if __name__ == '__main__':
    args = get_args()
    generate_query(args.input_file, args.output_file)

