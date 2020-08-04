

import tensorflow as tf

trained_checkpoint_prefix = 'saved_models/model.ckpt-49491'

checkpoint_dir = "saved_models"

# Construct a basic model.
latest = tf.train.load_checkpoint(checkpoint_dir)




# Save the model in SavedModel format.
export_dir = "saved_models"
input_data = tf.constant(1., shape=[1, 1])
to_save = root.f.get_concrete_function(input_data)
tf.saved_model.save(root, export_dir, to_save)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
   f.write(tflite_model)

