import io
import os
import json

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from PIL.ImageOps import fit, grayscale

MNIST_MODEL = load_model('./mnist.h5')
print(MNIST_MODEL.summary())

def post_image(file):
    image = Image.open(io.BytesIO(file.read()))
    image = grayscale(fit(image, (28, 28)))
    image_bytes = image.tobytes()
    image_array = np.reshape(np.frombuffer(image_bytes, dtype=np.uint8), (1, 28, 28, 1))
    prediction = MNIST_MODEL.predict(image_array)
    digit = np.argmax(prediction[0])
    return json.dumps({'digit':int(digit)})


