import time
import psutil
import tensorflow as tf
import tensorflow_hub as hub
from cpuinfo import get_cpu_info  # py-cpuinfo


def print_device_info():
    ####################################
    print("TF:", tf.__version__)
    ####################################
    print("Hub version:", hub.__version__)
    ####################################
    if tf.config.list_physical_devices('GPU'):
        local_device_protos = tf.python.client.device_lib.list_local_devices()
        gpu_details = [
            x for x in local_device_protos if x.device_type == 'GPU'][0]
        gpu_name = ''
        for d in gpu_details.physical_device_desc.split(', '):
            if d.split(':')[0] == 'name':
                gpu_name = d.split(': ')[1]
                break
        print(f"GPU: {gpu_name}")
    else:
        print("GPU: NOT AVAILABLE")
    ####################################
    cpu_info = get_cpu_info()
    print(f"CPU: {cpu_info['count']}x {cpu_info['brand']}")


def print_colab_time():
    uptime = time.time() - psutil.boot_time()
    remain = 12 * 60 * 60 - uptime
    print(time.strftime('%H:%M:%S', time.gmtime(remain)))


def save_class_indices(class_indices: dict, output_path: str):
    """Save class indices generated by keras ImageDataGenerator"""
    class_dict = class_indices
    class_list = sorted(class_dict.keys(), key=lambda x: class_dict[x])
    class_list_str = '\n'.join(list(class_list))
    with open(output_path, 'w') as f:
        f.write(class_list_str)


def load_class_indices(input_path: str):
    class_dict = {}
    with open(input_path, 'r') as f:
        for (i, label) in enumerate(f.read().split('\n')):
            class_dict[label] = i
    return class_dict
