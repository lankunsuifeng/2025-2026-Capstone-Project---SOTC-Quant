# src/gpu_utils.py
"""
GPU utility functions for TensorFlow/Keras training
Handles GPU detection, configuration, and memory management
"""
import tensorflow as tf
import os


def setup_gpu(memory_growth=True, gpu_id=None):
    """
    配置GPU设置
    
    Parameters:
    -----------
    memory_growth : bool
        是否启用GPU内存增长（避免一次性占用所有GPU内存）
    gpu_id : int or None
        指定使用的GPU ID（None表示使用所有可用GPU）
    
    Returns:
    --------
    dict : GPU信息字典，包含：
        - available: bool, GPU是否可用
        - device_name: str, GPU设备名称
        - memory_info: str, GPU内存信息
        - gpu_list: list, 可用GPU列表
    """
    gpu_info = {
        'available': False,
        'device_name': 'CPU',
        'memory_info': 'N/A',
        'gpu_list': []
    }
    
    # 检查GPU是否可用
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) == 0:
        return gpu_info
    
    gpu_info['available'] = True
    gpu_info['gpu_list'] = [gpu.name for gpu in gpus]
    
    try:
        if gpu_id is not None and gpu_id < len(gpus):
            # 只使用指定的GPU
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            gpu_info['device_name'] = gpus[gpu_id].name
        else:
            # 使用所有GPU
            tf.config.set_visible_devices(gpus, 'GPU')
            gpu_info['device_name'] = f"{len(gpus)} GPU(s)"
        
        # 配置GPU内存增长
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # 获取GPU详细信息
        if len(gpus) > 0:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            gpu_info['memory_info'] = f"Memory growth: {memory_growth}"
            if 'device_name' in gpu_details:
                gpu_info['device_name'] = gpu_details['device_name']
        
    except RuntimeError as e:
        # GPU配置必须在程序启动时设置，如果已经初始化会报错
        print(f"Warning: GPU configuration failed: {e}")
        gpu_info['available'] = False
    
    return gpu_info


def get_gpu_info():
    """
    获取当前GPU信息（不修改配置）
    
    Returns:
    --------
    dict : GPU信息字典
    """
    gpu_info = {
        'available': False,
        'device_name': 'CPU',
        'memory_info': 'N/A',
        'gpu_list': [],
        'tensorflow_version': tf.__version__
    }
    
    # 检查GPU是否可用
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 0:
        gpu_info['available'] = True
        gpu_info['gpu_list'] = [gpu.name for gpu in gpus]
        
        try:
            # 获取第一个GPU的详细信息
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if 'device_name' in gpu_details:
                gpu_info['device_name'] = gpu_details['device_name']
        except:
            gpu_info['device_name'] = gpus[0].name
        
        # 检查GPU是否在TensorFlow中可用
        if tf.test.is_gpu_available():
            gpu_info['memory_info'] = "GPU is available and configured"
        else:
            gpu_info['memory_info'] = "GPU detected but not available in TensorFlow"
    else:
        gpu_info['memory_info'] = "No GPU detected"
    
    return gpu_info


def print_gpu_info():
    """
    打印GPU信息到控制台
    """
    info = get_gpu_info()
    print("=" * 50)
    print("GPU Configuration")
    print("=" * 50)
    print(f"TensorFlow Version: {info['tensorflow_version']}")
    print(f"GPU Available: {info['available']}")
    print(f"Device Name: {info['device_name']}")
    print(f"GPU List: {info['gpu_list']}")
    print(f"Memory Info: {info['memory_info']}")
    print("=" * 50)
    
    # 打印所有GPU设备
    if info['available']:
        print("\nAvailable GPU Devices:")
        for i, gpu in enumerate(info['gpu_list']):
            print(f"  [{i}] {gpu}")
    
    return info
