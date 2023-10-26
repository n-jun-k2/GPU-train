import torch
from torchvision.transforms import v2, functional
from torchvision.utils import save_image
from const import VALUES

# Image Classification
CH, H, W = 3, 32, 32
img = torch.randint(0, 256, size=(CH, H, W), dtype=torch.uint8)
save_image(img, "img_pil.jpg")


transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transforms(img)
print(type(img)) # <class 'torch.Tensor'>
print(img.shape)

save_image(img, "random_image_pil.jpg")

# random_image_pil = functional.to_pil_image(img)
# random_image_pil.save("random_image_pil.jpg")

# グレースケール
# random_image_pil_gray = random_image_pil.convert("L")
# random_image_pil_gray.save("random_image_pil_gray.jpg")
