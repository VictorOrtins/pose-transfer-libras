from models.pose_transfer_model_inference import PoseTransferModelInference
#import torch
import cv2

model = PoseTransferModelInference(keypoints_numbers=274)

#state_dict = torch.load("./output/Libras/ckpt/2025-06-22-20-10-33/netG_20000.pth", map_location='cpu')
#state_dict = add_module_prefix(state_dict)
#model.netG.load_state_dict(state_dict)

image_condition_size = []
image_transfer_size = []

img = model.transfer_as('./frame_00001.jpg', 
                        './Aula_1_recorte_frame_00001.json', 
                        './_G0MZFLIHa0_9-5-rgb_front_000000000517_keypoints.json',
                        image_condition_size,
                       image_transfer_size)
#[342, 608]

cv2.imwrite(f"result.jpg", img)