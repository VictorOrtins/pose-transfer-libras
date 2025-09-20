import sys
import pandas as pd
from tqdm import tqdm
import cv2
import os

STEP = 5

output_dir = "frames_videos_teste"
csv_path = "./"
csv_name = "csv_pairs_test"

os.makedirs(output_dir, exist_ok=True)

video_paths = sys.argv[1]

dict_to_frame = {'file_name': [], 'frame_numbers': []}

for video_name in tqdm(os.listdir(video_paths)):


    cap = cv2.VideoCapture(f"{video_paths}/{video_name}")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{video_name.split('.mp4')[0]}_{frame_idx:012d}.jpg"
        cv2.imwrite(f"{output_dir}/{frame_name}", frame)
        frame_idx += 1

    dict_to_frame["file_name"].append(video_name.split('.mp4')[0])
    dict_to_frame["frame_numbers"].append(frame_idx + 1)
    cap.release()


dict_to_frame_pairs = {"imgA": [], "imgB": [], "video_name": []}

for video_name, total_frames in tqdm(zip(dict_to_frame["file_name"], dict_to_frame["frame_numbers"])):
    for i in range(0, total_frames - STEP - 2, STEP):
        input_frame = f"{video_name}_{i:012d}"
        target_frame = f"{video_name}_{i+STEP:012d}"

        dict_to_frame_pairs["imgA"].append(input_frame)
        dict_to_frame_pairs["imgB"].append(target_frame)
        dict_to_frame_pairs["video_name"].append(video_name)

df = pd.DataFrame(dict_to_frame_pairs)
df.to_csv(f"{csv_path}/{csv_name}.csv", index=False)