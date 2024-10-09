import base64
import requests
import cv2
import numpy as np
import openai
import json

try:
    api_key = open('robomimic/state_infuse/configs/openai.key', 'r').read().replace('\n', "")
except FileNotFoundError as e:
    api_key = open('configs/openai.key', 'r').read().replace('\n', "")


openai.api_key = api_key

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def change_ndarray_to_base64(image):
    # Convert the image to base64
    image = np.transpose(image, (1, 2, 0))
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str


def get_internal_state_form_openai(image_left, image_hand, image_right, complete_rate, task):
    base64_image_left = change_ndarray_to_base64(image_left)
    base64_image_hand = change_ndarray_to_base64(image_hand)
    base64_image_right = change_ndarray_to_base64(image_right)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": open('robomimic/state_infuse/configs/prompts.txt', 'r').read().replace('\n', '').format(
                            **{'task': task, 'complete_rate': 100 * complete_rate})
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_left}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_hand}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_right}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 1.0,
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return None


# Function to get embeddings for multiple strings
def get_embeddings(strings):
    model = "text-embedding-ada-002"
    embeddings = []
    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for string in strings:
        print(f"Getting embedding for: {string}")
        data = {
            "model": model,
            "input": string
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            embeddings.append(np.array(embedding))
        else:
            print(f"Error: {response.status_code} - {response.text}")
            embeddings.append(None)
    return embeddings


if __name__ == '__main__':
    # Call the function and get embeddings
    # List of strings you want to get embeddings for
    strings = [
        "The quick brown fox jumps over the lazy dog.",
    ]

    # Choose the embedding model, e.g., "text-embedding-ada-002"

    embeddings = get_embeddings(strings)

    # Print the embedding for each string
    for i, embedding in enumerate(embeddings):
        print(f"Embedding for string {i + 1}: {embedding[:5]}... (truncated for brevity)")