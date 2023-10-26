import torch
from torchvision import datasets
from const import PATH

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="imagenet-1k")

# imagenet_data = datasets.ImageNet(PATH.DATA_ROOT_DIR)
# data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True)