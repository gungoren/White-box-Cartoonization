try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import os
import cv2
import numpy as np
from test_code import network
from test_code import guided_filter
from tqdm import tqdm
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.saved_model import tag_constants


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    

def cartoonize(load_folder, save_folder, model_path):
    try:
        tf.disable_eager_execution()
    except:
        None

    tf.reset_default_graph()
    size = 512
    input_photo = tf.placeholder(tf.float32, [1, size, size, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    """
    model_dir = model_path
    output_graph_filename = os.path.join(model_dir, 'frozen_model.pb')
    initializer_nodes = ''
    freeze_graph(
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        output_node_names=output_node_names,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=output_graph_filename,
        clear_devices=True,
        initializer_nodes=initializer_nodes,
        input_meta_graph=False,
        input_saved_model_dir=model_dir,
        saved_model_tags=tag_constants.SERVING)
    """


    tf.train.write_graph(tf.get_default_graph(), model_path, 'saved_model.pb', as_text=False)
    tf.train.write_graph(tf.get_default_graph(), model_path, 'saved_model.pbtxt', as_text=True)

    gf = tf.GraphDef()
    gf.ParseFromString(open(os.path.join(model_path, 'saved_model.pb'), 'rb').read())

    nodes = [n.name + '=>' + n.op for n in gf.node if n.op in ('Softmax', 'Placeholder')]
    print(nodes)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_session(sess, [input_photo], [final_out]) # -- 7.6MB
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # -- 3.7 MB

    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("tflite model saved")
    

if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
