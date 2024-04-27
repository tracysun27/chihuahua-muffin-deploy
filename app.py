# commands to run
# export FLASK_APP=flask_test
# OPENCV_AVFOUNDATION_SKIP_AUTH=1
# export FLASK_ENV=development
# flask run

# for docker, if terminal gives u port 5000 daemon occupied or whatever try going to settings and turning off airplay receiver. idk why that works but it does.

from flask import Flask, render_template, Response, session, request, jsonify
from PIL import Image
from lru import LRU
import os
import cv2
import datetime as dt
import numpy as np
import tensorflow as tf
import base64
from rembg import remove

app = Flask(__name__)
# print("App Root Path:", app.root_path)
# print("Static Folder Path:", app.static_folder)

# # version 2: Images folder path
# app.config['UPLOAD_FOLDER'] = './static'
# # Check if the folder directory exists, if not then create it
# if not os.path.exists(app.config['UPLOAD_FOLDER'] ):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

model = tf.saved_model.load("./model_softmax_no_bg3")
images = LRU(32)


# Function to capture and save an image
def save_image(image_bytes):
    # Convert the bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image from the numpy array
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # making it square
    (height, width) = frame.shape[:2]
    if height < width:
        frame = frame[
            0:height, ((width // 2) - (height // 2)) : ((width // 2) + (height // 2))
        ]
    else:
        frame = frame[
            ((height // 2) - (width // 2)) : ((height // 2) + (width // 2)), 0:width
        ]

    # Construct the image name
    img_name = f'captured_image_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'

    # Save the image to the dict
    images[img_name] = frame

    return img_name


# # OLD: function for converting captured image into correct dimensions and tensor for model
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((32, 32))  # Resize the image to match the input size expected by the model
#     #img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     img = tf.convert_to_tensor(img, dtype=tf.float32)
#     return img


def preprocess_image(img_name):
    img = Image.fromarray(images[img_name])

    img = remove(img)
    img = img.convert("RGB")

    img = img.resize(
        (32, 32)
    )  # Resize the image to match the input size expected by the model
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

    # Add batch dimension and reshape to match model input shape
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img


def run_model(img_name, class_names=["chihuahua", "muffin"]):
    preprocessed_img = preprocess_image(img_name)
    classification = model(preprocessed_img)
    if (np.isclose(classification[0][0], 0.56, atol=0.01)) and (
        classification[0][1] >= 0.5
    ):
        index = 1
    else:
        index = np.argmax(classification)
    # index = np.argmax(res1)
    # plt.imshow(tf.keras.utils.load_img(img_path))
    return class_names[index]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/capture", methods=["POST"])
def capture():
    try:
        image_data = request.json["image"]
        image_data = image_data.split(",")[1]  # Remove the Base64 prefix
        image_bytes = base64.b64decode(image_data)
        img_name = save_image(image_bytes)  # Make sure save_image function is defined
        # Return the image name and path in the response
        return jsonify({"img_name": img_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results_page")
def results():
    img_name = request.args.get("img_name")
    res = run_model(img_name)
    base64_img = base64.b64encode(cv2.imencode(".jpg", images[img_name])[1]).decode()
    return render_template("results.html", base64_img=base64_img, res=res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
