from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

def predict_disease(image_path):
    # Load Yolo model
    model = YOLO('./RUN/best.pt')

    with Image.open(image_path) as img:
        img = img.resize((255, 255))

    results = model(img, show=True)

    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    prediction = names_dict[probs.index(max(probs))]

    return prediction

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route("/prediction", methods=['POST'])
def prediction():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    prediction = predict_disease(image_path)

    return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=3000, debug=True)