import torchvision
import torch
from model import MedPose
from utils import load_train
import time

# modify scatter and parallel_apply function 
# in data parallel for batch processing
import torch.nn.parallel.scatter_gather as scatter_gather
from torch.nn.parallel._functions import Scatter, Gather
import threading
from torch.cuda._utils import _get_device_index

DEVICES = [0, 1]

def custom_scatter(inputs, target_gpus, dim=0):

    def scatter_map(obj):
        # if isinstance(obj, torch.Tensor):
        #     print(obj.shape, "TENSOR SPLITTING TIME")
        #     return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            obj = obj[0]
        if isinstance(obj, list) and len(obj) > 0:
            if isinstance(obj[0], list) and len(obj) > 0:
                total_size = len(obj)
                scatter_size = int(total_size / len(target_gpus))
                scatter_out = []
                start = 0
                for target in target_gpus:
                    if start + scatter_size > total_size:
                        break
                    scatter_list = obj[start:(start + scatter_size)]
                    updated_scatter_list = []
                    for frame_batch in scatter_list:
                        for idx in range(len(frame_batch)):
                            frame_batch[idx] = frame_batch[idx].to(target)
                        updated_scatter_list.append(frame_batch)
                    scatter_out.append([updated_scatter_list])
                    start = start + scatter_size
                return scatter_out
        # if isinstance(obj, dict) and len(obj) > 0:
        #     return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None

    return res

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, valid_dataloader = load_train(batch_size=2, device=device)

window_size = 5

# set scatter function to custom scatter function
#model = torch.nn.DataParallel(MedPose(window_size=window_size)).to(device)
model = MedPose(window_size=window_size).to(device)
#model_replicas = torch.nn.parallel.replicate(MedPose(window_size=window_size).to(device), DEVICES)

def train():
    #global model_replicas
    for train_idx, (batch_videos, batch_keypoints) in enumerate(train_dataloader):
        print("=======================================")
        print("Processing {} videos...".format(len(batch_videos)))
        # scattered_batch_videos = custom_scatter(batch_videos, DEVICES)
        # print(len(scattered_batch_videos), len(scattered_batch_videos[0]), len(scattered_batch_videos[0][0]))
        # model_replicas = model_replicas[:len(scattered_batch_videos)]
        # print("applying parallelization...")
        # outs = torch.nn.parallel.parallel_apply(model_replicas, scattered_batch_videos)
        # print("gathering all outputs...")
        # out = torch.nn.parallel.gather(outs, DEVICES[0])
        out = model(batch_videos)
        del out
        print("=======================================")
        # for frames in batch_videos:
        #     x = []
        #     pose_detections = []
        #     for fidx, frame in enumerate(frames):
        #         print("=======================================")
        #         x.append(frame.to(device))
        #         x = x[-window_size:]
        #         print("Processing {} frames...".format(len(x)))
        #         print("Iteration {}...".format(fidx + 1))
        #         initial_frame = (len(x) == 1)
        #         print("Initial frame status: ", initial_frame)
        #         print(x[0].shape)
        #         out = model(x, pose_detections, initial_frame=initial_frame)
        #         pose_detections.append(out)
        #         pose_detections = pose_detections[-window_size:]
        #         del out
        #         print("currently produced", len(pose_detections), "pose detections")
        #         print("=======================================")

if __name__ == '__main__':
    print("[III] Training MedPose...")
    #for model in model_replicas:
    print("Number of trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), "on GPU:", next(model.parameters()).get_device())
    train()


