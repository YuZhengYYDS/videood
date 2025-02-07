import os
import io
import torch
import numpy as np
from tqdm import tqdm

try:
    from petrel_client.client import Client
except ImportError:
    Client = None

class checkpoint_ceph(object):
    def __init__(self, conf_path="~/petreloss.conf", checkpoint_dir="weatherbench:s3://weatherbench/checkpoint") -> None:
        # 如果未能导入 Client 或 conf_path 为空，则不启用 ceph 操作
        if Client is None or not conf_path:
            self.client = None
        else:
            self.client = Client(conf_path=conf_path)
        self.checkpoint_dir = checkpoint_dir

    def load_checkpoint(self, url):
        url = os.path.join(self.checkpoint_dir, url)
        if self.client is None or not self.client.contains(url):
            return None
        with io.BytesIO(self.client.get(url, update_cache=True)) as f:
            checkpoint_data = torch.load(f, map_location=torch.device('cpu')) 
        return checkpoint_data
    
    def load_checkpoint_with_ckptDir(self, url, ckpt_dir):
        url = os.path.join(ckpt_dir, url)
        if self.client is None or not self.client.contains(url):
            return None
        with io.BytesIO(self.client.get(url, update_cache=True)) as f:
            checkpoint_data = torch.load(f, map_location=torch.device('cpu')) 
        return checkpoint_data
    
    def save_checkpoint(self, url, data):
        url = os.path.join(self.checkpoint_dir, url)
        if self.client is None:
            # 若无客户端，则直接保存到本地文件
            torch.save(data, url)
        else:
            with io.BytesIO() as f:
                torch.save(data, f)
                f.seek(0)
                self.client.put(url, f)

    def save_prediction_results(self, url, data):
        url = os.path.join(self.checkpoint_dir, url)
        if self.client is None:
            np.save(url, data)
        else:
            with io.BytesIO() as f:
                np.save(f, data)
                f.seek(0)
                self.client.put(url, f)
