from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename 
# from st import image_preparer, get_model_and_losses, run_style_transfer
import os
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image


app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the post request has the file part
    if 'file1' not in request.files or 'file2' not in request.files:
        return redirect(request.url)

    file1 = request.files['file1']
    file2 = request.files['file2']

    # If user does not select file, browser also submit an empty part without filename
    if file1.filename == '' or file2.filename == '':
        return redirect(request.url)

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        # Save the uploaded files
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        # Process the images and create a result image (you need to implement this part)
        result_filename = process_images(filename1, filename2)

        # Return the result image
        return redirect(url_for('result', filename=result_filename))

    return redirect(request.url)

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def process_images(filename1, filename2):
    from PIL import Image

    content_img = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
    style_img = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

    print('Model loading ...')
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    styled_image = hub_model(tf.constant(content_img), tf.constant(style_img))[0]
    styled_image = tensor_to_image(styled_image)

    result_filename = f'{filename1}_{filename2}'
    save_in = os.path.join('./static', result_filename)
    styled_image.save(save_in)

    print('image saved successfuly')
    return result_filename

if __name__ == '__main__':
    app.run(debug=True)
