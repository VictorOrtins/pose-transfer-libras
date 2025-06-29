import os
import re
import cv2
import torch
from models.pose_transfer_model_inference import PoseTransferModelInference

# Configurações
condition_img_path = './dataset/frames_videos_teste/_fZbAxSSbX4_0-5-rgb_front_000000000000.jpg'
condition_pose_json_path = './dataset/openpose_output_teste/json/_fZbAxSSbX4_0-5-rgb_front/_fZbAxSSbX4_0-5-rgb_front_000000000000_keypoints.json'
transfer_pose_dir_path = './Aula 1 - Recorte/Aula_1_recorte'
image_condition_size = []
image_transfer_size = []
output_video_path = 'video_resultado.mp4'
fps = 30

model = PoseTransferModelInference(keypoints_numbers=274)
state_dict = torch.load("./output/Libras/ckpt/2025-06-22-20-10-33/netG_40000.pth", map_location='cpu')
model.netG.load_state_dict(state_dict)

def extract_frame_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

json_files = sorted(
    [f for f in os.listdir(transfer_pose_dir_path) if f.endswith('.json')],
    key=extract_frame_number
)

# Inicializar vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, tuple([256, 256]))

frame_count = 1
for json_file in json_files:
    transfer_json_path = os.path.join(transfer_pose_dir_path, json_file)

    img_result = model.transfer_as(
        condition_img_path,
        condition_pose_json_path,
        transfer_json_path,
        image_condition_size,
        image_transfer_size
    )

    video_writer.write(img_result)
    print(frame_count)
    frame_count += 1

video_writer.release()
print(f"Vídeo salvo com sucesso: {output_video_path}")
