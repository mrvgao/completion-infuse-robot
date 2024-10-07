import base64
import requests
import cv2

api_key = "sk-..."

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def change_ndarray_to_base64(image):
    # Convert the image to base64
    _, buffer = cv2.imencode('.jpg', image)
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
              "text": f"there three images are left, mid, and right images of a robot who is executing the task {task} at "
                      f"step {step} out of {horizon}. The {step}/{horizon} can be treat as completion rate of the task. Please specify"
                      f"what actions of this robots here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please"
                      f"give what the robot should do to avoid the error."
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
      "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())