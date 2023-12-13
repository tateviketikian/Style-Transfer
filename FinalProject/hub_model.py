import tensorflow_hub as hub
import IPython.display as display
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


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

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(60)



if __name__ == "__main__":
    # print(tf.config.list_physical_devices('GPU'))
    tf.debugging.set_log_device_placement(False)
    content_path = './test_images/content/dancing.jpg'
    style_path = './test_images/styles/picasso_selfport1907.jpg'

    content_name = os.path.basename(content_path).split('.')[0]
    style_name = os.path.basename(style_path)

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    start_time = time.time()
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model load time: {elapsed_time} seconds")

    styled_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    styled_img = tensor_to_image(styled_image)
    save_in = os.path.join('./static', f'{content_name}_{style_name}')
    styled_img.save(save_in)

    styled_img = load_img(save_in)
    imshow(styled_img)

