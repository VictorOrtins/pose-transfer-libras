import torch
from .netG import NetG2
import cv2
import numpy as np
import json
from PIL import Image
import torchvision.transforms as T

class PoseTransferModelInference():
    
    def __init__(self, keypoints_numbers = 36):
        super(PoseTransferModelInference, self).__init__()

        self.netG = NetG2(3, keypoints_numbers, 3)
        self.netG.eval()

        self._img_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transform_map = T.Compose([
            T.ToTensor()
        ])

    def get_keypoints_heatmap(self, json_data, img_shape, size=256, parts=["pose", "face", "hand_left", "hand_right"]):
        """
        Gera heatmaps e weightmap sem filtro gaussiano — usa quadrados ao redor dos keypoints.

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

        for i, (x, y, conf) in enumerate(keypoints_data):
            if conf < 0.1 or x <= 0 or y <= 0:
                continue
            x_int = int(np.clip(x * size / img_shape[0], 0, size - 1))  
            y_int = int(np.clip(y * size / img_shape[1], 0, size - 1))
            heatmaps[y_int, x_int, i] = 1.0

        return heatmaps
    
    def _load_image(self, image_path):
        imgA = Image.open(f"{image_path}")

        return self._img_transform(imgA)

    def transfer_as(self, condition_image_path, 
                    path_pose_condition,
                    path_pose_transfer,
                    shape_img_condition,
                    shape_img_trasfer):

        mapA_t = self.get_keypoints_heatmap(json.load(open(f"{path_pose_condition}")), shape_img_condition)
        mapB_t = self.get_keypoints_heatmap(json.load(open(f"{path_pose_transfer}")), shape_img_trasfer)

        imgA_t = self._load_image(condition_image_path).unsqueeze(0)

        mapA_t = self.transform_map(mapA_t).unsqueeze(0)
        mapB_t = self.transform_map(mapB_t).unsqueeze(0)

        mapAB_t = torch.cat((mapA_t, mapB_t), dim=1)
        with torch.no_grad():
            imgA_t_trasfered = self.netG(imgA_t, mapAB_t).squeeze()

        print(imgA_t_trasfered.shape)

        imgA_t_trasfered = imgA_t_trasfered.permute(1, 2, 0).numpy()  # [H, W, C]
        imgA_t_trasfered = (imgA_t_trasfered + 1) / 2 
        imgA_t_trasfered = (imgA_t_trasfered * 255).astype(np.uint8)
        
        imgA_t_trasfered = cv2.cvtColor(imgA_t_trasfered, cv2.COLOR_RGB2BGR)
 
        return imgA_t_trasfered
