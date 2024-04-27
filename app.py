# commands to run
# export FLASK_APP=flask_test
# OPENCV_AVFOUNDATION_SKIP_AUTH=1
# export FLASK_ENV=development
# flask run

# for docker, if terminal gives u port 5000 daemon occupied or whatever try going to settings and turning off airplay receiver. idk why that works but it does.

from flask import Flask, render_template, Response, session, request, jsonify
from PIL import Image
import os
import cv2
import datetime as dt
import numpy as np
import tensorflow as tf
import base64
from rembg import remove

app = Flask(__name__)
app.secret_key = "509d5ad1fb502054034e60c79aa439b3"
# print("App Root Path:", app.root_path)
# print("Static Folder Path:", app.static_folder)

# version 1: save in desktop folder named "images"
try:
    os.mkdir("./static")
except OSError as error:
    pass

# # version 2: Images folder path
# app.config['UPLOAD_FOLDER'] = './static'
# # Check if the folder directory exists, if not then create it
# if not os.path.exists(app.config['UPLOAD_FOLDER'] ):
#     os.makedirs(app.config['UPLOAD_FOLDER'])


def resize_image(height, width):
    min_dim = min(height, width)
    if min_dim == height:
        new_height = height
        new_width = width // 2
    else:
        new_height = height // 2
        new_width = width
    return new_height, new_width


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

    # Construct the image name and path
    img_name = f'captured_image_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
    img_path = f"./static/{img_name}"

    # Save the image to the specified path
    cv2.imwrite(img_path, frame)

    return img_name, img_path


# # OLD: function for converting captured image into correct dimensions and tensor for model
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((32, 32))  # Resize the image to match the input size expected by the model
#     #img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     img = tf.convert_to_tensor(img, dtype=tf.float32)
#     return img


def preprocess_image(image_path):
    img = Image.open(image_path)

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


# Function to generate frames from webcam feed
def generate_frames():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        else:
            # my webcam is 1280 by 720. make it square
            (height, width) = frame.shape[:2]
            if height < width:
                frame = frame[
                    0:height,
                    ((width // 2) - height // 2) : ((width // 2) + height // 2),
                ]
            else:
                frame = frame[
                    ((height // 2) - width // 2) : ((height // 2) + width // 2), 0:width
                ]
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def run_model(img_path, class_names=["chihuahua", "muffin"]):
    model = tf.saved_model.load("./model_softmax_no_bg3")
    test1 = preprocess_image(img_path)
    res1 = model(test1)
    if (np.isclose(res1[0][0], 0.56, atol=0.01)) and (res1[0][1] >= 0.5):
        index = 1
    else:
        index = np.argmax(res1)
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
        img_name, img_path = save_image(
            image_bytes
        )  # Make sure save_image function is defined
        # Return the image name and path in the response
        return jsonify({"img_name": img_name, "img_path": img_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/results_page")
def results():
    img_path = session.get("img_path", None)
    res = run_model(img_path)
    img_name = request.args.get("img_name")
    return render_template(
        "results.html", img_path=img_path, img_name=img_name, res=res
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
