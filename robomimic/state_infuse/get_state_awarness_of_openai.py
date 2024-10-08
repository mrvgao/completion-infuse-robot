import base64
import requests
import cv2
import numpy as np

api_key = "sk-..."


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


def get_internal_state_form_openai(image_left, image_hand, image_right, step, horizon, task):
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
                            **{'task': task, 'step': step, 'horizon': horizon})
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
        "max_tokens": 300,
        "temperature": 0.1,
        "top_p": 1.0,
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        import pdb; pdb.set_trace()

        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return None
