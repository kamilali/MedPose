import cv2
import os
import torchvision
import torch
from model import MedPose

test_img = cv2.imread("./data/test-frames/out-006.jpg")
test_img2 = cv2.imread("./data/test-frames/out-007.jpg")
test_img3 = cv2.imread("./data/test-frames/out-008.jpg")
test_img4 = cv2.imread("./data/test-frames/out-009.jpg")

test_img_tensor = torchvision.transforms.functional.to_tensor(test_img)
test_img2_tensor = torchvision.transforms.functional.to_tensor(test_img2)
test_img3_tensor = torchvision.transforms.functional.to_tensor(test_img3)
test_img4_tensor = torchvision.transforms.functional.to_tensor(test_img4)
video_frames = [test_img_tensor.float().cuda(), test_img2_tensor.float().cuda(), test_img3_tensor.float().cuda(), test_img4_tensor.float().cuda()]

#model = torch.nn.DataParallel(MedPose()).cuda()
model = MedPose().cuda()
model.train()

x = []
pose_detections = []

print("Number of trainable model parameters: ", sum(p.numel() for p in model.parameters()))
print("Number of trainable model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

for video_frame in video_frames:
    print("=======================================")
    '''
    pose detection for current frame (in model, pass
    in frames + pose detections)
    '''
    x.append(video_frame)
    print("Processing {} frames...".format(len(x)))
    out = model(x, pose_detections)
    pose_detections.append(out)
    print("currently produced", len(pose_detections), "pose detections")
    print("=======================================")
