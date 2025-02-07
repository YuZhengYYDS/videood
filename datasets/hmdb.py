import torch.utils.data
import imageio.v3 as iio
import numpy as np
import os
import torchvision.transforms as transforms

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().split() for line in lines if line.strip()]
    paths, labels = zip(*data)
    return paths, labels

class HMDBFrame(torch.utils.data.Dataset):
    def __init__(self, split='train_near_ood', eval=False, cfg=None, dataset='HMDB', datapath='/media/volume/HMDB51_video/video', image_size=(240, 320)):
        """
        Args:
            split (str): Split name, e.g., 'train_near_ood'
            eval (bool): Evaluation mode flag (not used here)
            cfg: Configuration object (not used now)
            dataset (str): Dataset name, used to build the split file name
            datapath (str): Base path for the videos (e.g., "/media/volume/HMDB51_video/video")
            image_size (tuple): The target size to resize frames
        """
        self.base_path = datapath
        self.split = split
        self.cfg = cfg
        self.dataset = dataset
        self.image_size = image_size  # 目标尺寸

        # 读取 split 文件
        split_file_name = os.path.join(os.path.dirname(__file__), "splits", f"{dataset}_{split}.txt")
        self.samples, self.labels = load_txt_file(split_file_name)

        # 生成索引映射
        self.index_map = []
        for sample in self.samples:
            video_file = os.path.join(self.base_path, sample)
            try:
                vid = iio.imread(video_file, plugin="pyav")
            except Exception as e:
                print(f"Error loading video {video_file}: {e}")
                continue
            num_frames = vid.shape[0]
            for frame_idx in range(num_frames):
                self.index_map.append((video_file, frame_idx))
        print(f"Total frames in dataset: {len(self.index_map)}")

        # 定义 transform：统一调整大小
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor()  # 归一化到 [0, 1]
        ])

    def __getitem__(self, index):
        video_file, frame_idx = self.index_map[index]
        # 读取视频并提取指定帧
        vid = iio.imread(video_file, plugin="pyav")
        frame = vid[frame_idx]  # 提取该帧

        # 转换为统一大小
        frame = self.transform(frame)

        # 构造返回数据
        data = {
            'inputs': frame.clone().detach(), # 直接转换为 PyTorch 张量
            'data_samples': frame.clone().detach(),  # 作为目标数据
            'video_file': video_file,
            'frame_idx': frame_idx
        }
        return data

    def __len__(self):
        return len(self.index_map)
