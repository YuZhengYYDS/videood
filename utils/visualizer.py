import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import io

try:
    from petrel_client.client import Client
except ImportError:
    Client = None

class non_visualizer:
    """占位符可视化器 (适用于 HMDB 或无可视化需求的情况)"""
    def save_pixel_image(self, *args, **kwargs):
        pass

    def save_vil_image(self, *args, **kwargs):
        pass

    def save_npy(self, *args, **kwargs):
        pass

# 适配不同数据集类型的可视化器
def get_visualizer(exp_dir, dataset_type="hmdb"):
    return non_visualizer()
