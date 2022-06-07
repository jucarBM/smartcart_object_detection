import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
input_saved_model_dir = "models/fine_tuned_model_5000_ds/saved_model"
output_saved_model_dir = "models/converted"
converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
converter.convert()
converter.save(output_saved_model_dir)
