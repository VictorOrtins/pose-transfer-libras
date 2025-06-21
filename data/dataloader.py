import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import json


class PoseDataset(Dataset):
    
    def __init__(self, dataset_dir, img_pairs, pose_maps_dir,
                 img_transform=None, map_transform=None, reverse=False):
        super(PoseDataset, self).__init__()
        self._dataset_dir = dataset_dir
        self._img_pairs = pd.read_csv(img_pairs)
        self._pose_maps_dir = pose_maps_dir
        self._img_transform = img_transform or transforms.ToTensor()
        self._map_transform = map_transform or transforms.ToTensor()
        self._reverse = reverse
    
    def __len__(self):
        return len(self._img_pairs)
    
    def __getitem__(self, index):
        pthA = self._img_pairs.iloc[index].imgA
        pthB = self._img_pairs.iloc[index].imgB
        
        fidA = os.path.splitext(pthA)[0].replace('/', '').replace('\\', '')
        fidB = os.path.splitext(pthB)[0].replace('/', '').replace('\\', '')
        
        imgA = Image.open(os.path.join(self._dataset_dir, pthA))
        imgB = Image.open(os.path.join(self._dataset_dir, pthB))
        
        mapA = np.float32(np.load(os.path.join(self._pose_maps_dir, f'{fidA}.npz'))['arr_0'])
        mapB = np.float32(np.load(os.path.join(self._pose_maps_dir, f'{fidB}.npz'))['arr_0'])
        
        imgA = self._img_transform(imgA)
        imgB = self._img_transform(imgB)
        
        mapA = self._map_transform(mapA)
        mapB = self._map_transform(mapB)
        
        if not self._reverse:
            return {'imgA': imgA, 'imgB': imgB, 'mapA': mapA, 'mapB': mapB, 'fidA': fidA, 'fidB': fidB}
        else:
            return {'imgA': imgB, 'imgB': imgA, 'mapA': mapB, 'mapB': mapA, 'fidA': fidB, 'fidB': fidA}
        
class PoseDatasetV2(Dataset):
    
    def __init__(self, rgb_frames_dir, json_keypoints_dir, img_pairs,
                 img_transform=None, map_transform=None, reverse=False):
        super(PoseDataset, self).__init__()
        self._rgb_frames_dir = rgb_frames_dir
        self._json_keypoints_dir = json_keypoints_dir
        self._img_pairs = pd.read_csv(img_pairs)
        self._img_transform = img_transform or transforms.ToTensor()
        self._map_transform = map_transform or transforms.ToTensor()
        self._reverse = reverse
    
    def __len__(self):
        return len(self._img_pairs)

    def get_keypoints_heatmap(json_data, size=256, parts=["pose", "face", "hand_left", "hand_right"], sigma=3):
        """
        Converte os keypoints do JSON do OpenPose em heatmaps.
        
        Parâmetros:
            json_data: dicionário carregado do .json
            size: dimensão da imagem de saída (size x size)
            parts: quais partes incluir ["pose", "face", "hand_left", "hand_right"]
            sigma: desvio padrão do Gaussian para suavizar o ponto

        Retorno:
            heatmaps: array (size, size, num_keypoints)
        """

        people = json_data.get("people", [])
        if not people:
            return np.zeros((size, size, 1), dtype=np.float32)
        
        p = people[0] 
        keypoints_data = []

        if "pose" in parts:
            keypoints_data.extend(p["pose_keypoints_2d"])
        if "face" in parts:
            keypoints_data.extend(p["face_keypoints_2d"])
        if "hand_left" in parts:
            keypoints_data.extend(p["hand_left_keypoints_2d"])
        if "hand_right" in parts:
            keypoints_data.extend(p["hand_right_keypoints_2d"])

        keypoints_data = np.array(keypoints_data).reshape(-1, 3)  # (N, 3)
        num_keypoints = keypoints_data.shape[0]
        
        heatmaps = np.zeros((size, size, num_keypoints), dtype=np.float32)

        for i, (x, y, conf) in enumerate(keypoints_data):
            if conf < 0.1 or x <= 0 or y <= 0:
                continue
            x_int = int(np.clip(x * size / 1280, 0, size - 1))  
            y_int = int(np.clip(y * size / 720, 0, size - 1))
            heatmaps[y_int, x_int, i] = 1.0

        for i in range(num_keypoints):
            heatmaps[:, :, i] = gaussian_filter(heatmaps[:, :, i], sigma=sigma)

        return heatmaps

    
    def __getitem__(self, index):
        pthA = self._img_pairs.iloc[index].imgA
        pthB = self._img_pairs.iloc[index].imgB
        

        imgA = Image.open(f"{self._rgb_frames_dir}/{pthA}")
        imgB = Image.open(f"{self._rgb_frames_dir}/{pthB}")

        mapA = self.get_keypoints_heatmap(json.load(open(f"{self._json_keypoints_dir}/{pthA}")))
        mapB = self.get_keypoints_heatmap(json.load(open(f"{self._json_keypoints_dir}/{pthB}")))
        
        imgA = self._img_transform(imgA)
        imgB = self._img_transform(imgB)
        
        mapA = self._map_transform(mapA)
        mapB = self._map_transform(mapB)
        
        if not self._reverse:
            return {'imgA': imgA, 'imgB': imgB, 'mapA': mapA, 'mapB': mapB}
        else:
            return {'imgA': imgB, 'imgB': imgA, 'mapA': mapB, 'mapB': mapA}


def create_dataloader(dataset_dir, img_pairs, pose_maps_dir,
                      img_transform=None, map_transform=None, reverse=False,
                      batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    
    dataset = PoseDataset(dataset_dir, img_pairs, pose_maps_dir, img_transform, map_transform, reverse)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)

def create_dataloaderV2(rgb_frames_dir, json_keypoints_dir, img_pairs,
                      img_transform=None, map_transform=None, reverse=False,
                      batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    
    dataset = PoseDataset(rgb_frames_dir, json_keypoints_dir, img_pairs, img_transform, map_transform, reverse)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
