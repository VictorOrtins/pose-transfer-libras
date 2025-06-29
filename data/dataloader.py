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
        super(PoseDatasetV2, self).__init__()
        self._rgb_frames_dir = rgb_frames_dir
        self._json_keypoints_dir = json_keypoints_dir
        self._img_pairs = pd.read_csv(img_pairs)
        self._img_transform = img_transform or transforms.ToTensor()
        self._map_transform = map_transform or transforms.ToTensor()
        self._reverse = reverse
        self.weight_hand = 3.0
        self.weight_face = 2.0
        self.weight_pose = 1.0
        self.weight_another = 0.3

    
    def __len__(self):
        return len(self._img_pairs)

    def get_keypoints_heatmap_and_weightmap(self, json_data, size=256, parts=["pose", "face", "hand_left", "hand_right"], box_size=8):
        """
        Gera heatmaps e weightmap (usa quadrados ao redor dos keypoints).

        Retorna:
            heatmaps: (H, W, num_keypoints)
            weight_map: (H, W)
        """
        people = json_data.get("people", [])
        if not people:
            return np.zeros((size, size, 1), dtype=np.float32), np.ones((size, size), dtype=np.float32)

        p = people[0]
        keypoints_data = []
        part_labels = []

        if "pose" in parts:
            keypoints_data.extend(p["pose_keypoints_2d"])
            part_labels.extend(["pose"] * 25)
        if "face" in parts:
            keypoints_data.extend(p["face_keypoints_2d"])
            part_labels.extend(["face"] * 70)
        if "hand_left" in parts:
            keypoints_data.extend(p["hand_left_keypoints_2d"])
            part_labels.extend(["hand_left"] * 21)
        if "hand_right" in parts:
            keypoints_data.extend(p["hand_right_keypoints_2d"])
            part_labels.extend(["hand_right"] * 21)

        keypoints_data = np.array(keypoints_data).reshape(-1, 3)
        num_keypoints = keypoints_data.shape[0]

        heatmaps = np.zeros((size, size, num_keypoints), dtype=np.float32)
        weight_map = np.zeros((size, size), dtype=np.float32)

        for i, ((x, y, conf), part_type) in enumerate(zip(keypoints_data, part_labels)):
            if conf < 0.1 or x <= 0 or y <= 0:
                continue

            x_int = int(np.clip(x * size / 1280, 0, size - 1))
            y_int = int(np.clip(y * size / 720, 0, size - 1))

            heatmaps[y_int, x_int, i] = 1.0

            y_min = max(0, y_int - box_size)
            y_max = min(size, y_int + box_size)
            x_min = max(0, x_int - box_size)
            x_max = min(size, x_int + box_size)

            if "hand" in part_type:
                weight_map[y_min:y_max, x_min:x_max] += self.weight_hand
            elif "face" in part_type:
                weight_map[y_min:y_max, x_min:x_max] += self.weight_face
            elif "pose" in part_type:
                weight_map[y_min:y_max, x_min:x_max] += self.weight_pose
            else:
                weight_map[y_min:y_max, x_min:x_max] += self.weight_another

        return heatmaps, weight_map
    
    def get_keypoints_heatmap(self, json_data, size=256, parts=["pose", "face", "hand_left", "hand_right"]):
        """
        Converte os keypoints do JSON do OpenPose em heatmaps.
        
        Parâmetros:
            json_data: dicionário carregado do .json
            size: dimensão da imagem de saída (size x size)
            parts: quais partes incluir ["pose", "face", "hand_left", "hand_right"]
   
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

        return heatmaps

    
    def __getitem__(self, index):
        pthA = self._img_pairs.iloc[index].imgA
        pthB = self._img_pairs.iloc[index].imgB
        video_name = self._img_pairs.iloc[index].video_name
        

        imgA = Image.open(f"{self._rgb_frames_dir}/{pthA}.jpg")
        imgB = Image.open(f"{self._rgb_frames_dir}/{pthB}.jpg")

        mapA = self.get_keypoints_heatmap(json.load(open(f"{self._json_keypoints_dir}/{video_name}/{pthA}_keypoints.json")))
        mapB, WmapB = self.get_keypoints_heatmap_and_weightmap(json.load(open(f"{self._json_keypoints_dir}/{video_name}/{pthB}_keypoints.json")))
        
        imgA = self._img_transform(imgA)
        imgB = self._img_transform(imgB)
        
        mapA = self._map_transform(mapA)
        mapB = self._map_transform(mapB)
        WmapB = self._map_transform(WmapB)
        
        if not self._reverse:
            return {'imgA': imgA, 'imgB': imgB, 'mapA': mapA, 'mapB': mapB, 'WmapB': WmapB}
        else:
            return {'imgA': imgB, 'imgB': imgA, 'mapA': mapB, 'mapB': mapA, 'WmapB': WmapB}


def create_dataloader(dataset_dir, img_pairs, pose_maps_dir,
                      img_transform=None, map_transform=None, reverse=False,
                      batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    
    dataset = PoseDataset(dataset_dir, img_pairs, pose_maps_dir, img_transform, map_transform, reverse)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)

def create_dataloaderV2(rgb_frames_dir, json_keypoints_dir, img_pairs,
                      img_transform=None, map_transform=None, reverse=False,
                      batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    
    dataset = PoseDatasetV2(rgb_frames_dir, json_keypoints_dir, img_pairs, img_transform, map_transform, reverse)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
