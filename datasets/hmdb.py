import torch.utils.data
import imageio.v3 as iio
import numpy as np
import os

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().split() for line in lines if line.strip()]
    paths, labels = zip(*data)
    return paths, labels

class HMDBFrame(torch.utils.data.Dataset):
    def __init__(self, split='train_near_ood', eval=False, cfg=None, dataset='HMDB', datapath='/media/volume/HMDB51_video/video'):
        """
        Args:
            split (str): Split name, e.g., 'train_near_ood'
            eval (bool): Evaluation mode flag (not used here)
            cfg: Configuration object (not used now)
            dataset (str): Dataset name, used to build the split file name
            datapath (str): Base path for the videos (e.g., "/media/volume/HMDB51_video/video")
        """
        self.base_path = datapath
        self.split = split
        self.cfg = cfg
        self.dataset = dataset

        # Load the split file; expected file: "splits/HMDB_{split}.txt"
        split_file_name = os.path.join("splits", f"{dataset}_{split}.txt")
        self.samples, self.labels = load_txt_file(split_file_name)

        # Build a mapping from a global index to (video_file, frame_index)
        self.index_map = []
        for sample in self.samples:
            # Note: since datapath already points to the "video" folder, we simply join the sample.
            video_file = os.path.join(self.base_path, sample)
            try:
                # Read the video to determine the number of frames.
                vid = iio.imread(video_file, plugin="pyav")
            except Exception as e:
                print(f"Error loading video {video_file}: {e}")
                continue
            num_frames = vid.shape[0]
            for frame_idx in range(num_frames):
                self.index_map.append((video_file, frame_idx))
        print(f"Total frames in dataset: {len(self.index_map)}")

    def __getitem__(self, index):
        video_file, frame_idx = self.index_map[index]
        # Load the video and extract the specific frame.
        vid = iio.imread(video_file, plugin="pyav")
        frame = vid[frame_idx]  # Extract the desired frame
        
        # Prepare a data dictionary.
        data = {
            'frame': frame,         # The single frame as a NumPy array
            'video_file': video_file,
            'frame_idx': frame_idx
        }
        return data

    def __len__(self):
        return len(self.index_map)
