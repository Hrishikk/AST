from flask import Flask, render_template,request
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import PIL

from tensorflow.keras import Model

opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99, beta_2=0.999, epsilon=1e-1) #Adam optimizer-minimizes the loss function
app = Flask(__name__)


@app.route ("/")
def home():
    return render_template("index.html")
@app.route("/processimages", methods=["POST"])
def processimages():
    image1 = request.files["image1"]
    image2 = request.files["image2"]
# get the shape of the image array
    
    processed_images = process_image(image1.filename,image2.filename)
    image_folder = os.path.join('static','images')
    images = os.listdir(image_folder)
    return render_template('inner-page.html', images=images)

    
def process_image(image1,image2):
  def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    gram_matrix = tf.expand_dims(result, axis=0)
    input_shape = tf.shape(input_tensor)
    weightofij = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return gram_matrix/weightofij

  def load_vgg():
    vgg = tf.keras.applications.VGG19(include_top=True, weights=None)
    vgg.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5')
    vgg.trainable = False
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_output = vgg.get_layer(content_layers[0]).output 
    style_output = [vgg.get_layer(style_layer).output for style_layer in style_layers]
    gram_style_output = [gram_matrix(output_) for output_ in style_output]

    model = Model([vgg.input], [content_output, gram_style_output])
    return model

  def loss_object(style_outputs, content_outputs, style_target, content_target):
    style_weight = 1e-2
    content_weight = 1e-1
    content_loss = tf.reduce_mean((content_outputs - content_target)**2) #formula of content loss 
    style_loss = tf.add_n([tf.reduce_mean((output_ - target_)**2) for output_, target_ in zip(style_outputs, style_target)]) #formula of styleloss 
    total_loss = content_weight*content_loss + style_weight*style_loss #formula of total loss 
    return total_loss

  def train_model(image, epoch,vgg_model):
    with tf.GradientTape() as tape:
      output = vgg_model(image*255)
      loss = loss_object(output[1], output[0], style_target, content_target)
    gradient = tape.gradient(loss, image)
    opt.apply_gradients([(gradient, image)]) #to optimize the loss and back propogation
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
  #tf.print(f"Loss = {loss} epoch={epoch}")
    tensor = image*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    tensor =  PIL.Image.fromarray(tensor)
    plt.imshow(cv2.cvtColor(np.array(tensor), cv2.COLOR_BGR2RGB))
    plt.savefig(f'static/images/image_{epoch}.png')

  content_image = cv2.resize(cv2.imread(image1), (224, 224))
  content_image = tf.image.convert_image_dtype(content_image, tf.float32)
  style_image = cv2.resize(cv2.imread(image2), (224, 224))
  style_image = tf.image.convert_image_dtype(style_image, tf.float32)

  vgg_model = load_vgg()
  global content_target
  global style_target
  content_target = vgg_model(np.array([content_image*224]))[0]
  style_target = vgg_model(np.array([style_image*224]))[1]
  
  EPOCHS = 30
  image = tf.image.convert_image_dtype(content_image, tf.float32)
  image = tf.Variable([image])
  for i in range(0,EPOCHS,10):
    train_model(image, i,vgg_model)

if __name__ == "__main__":
      app.run(debug=False,host='0.0.0.0')
