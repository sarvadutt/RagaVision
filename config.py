# config.py
import os
import tensorflow as tf

def configure_environment():
    # Isolate TF from other ML frameworks
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    
    # Disable parallel optimizations
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Prevent TF from claiming all GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)