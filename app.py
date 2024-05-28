from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import io

app = Flask(__name__)

# Setup the DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            image = Image.open(file.stream)
            return process_image(image)
        else:
            return "File type not allowed. File type should be png, jpg, or jpeg"
    return render_template('index.html')

def process_image(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detection_results = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detection_results.append(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    return "<br>".join(detection_results)

if __name__ == '__main__':
    app.run(debug=True)
