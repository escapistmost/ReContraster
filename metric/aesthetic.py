import os
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
import clip
from torch import nn

class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = nn.Linear(feats_in, 1)

    def forward(self, input):
        x = F.normalize(input, dim=-1) * input.shape[-1] ** 0.5
        return self.linear(x)


# 获取所有文件路径
def get_filepaths(parentpath, filepaths):
    paths = []
    for path in filepaths:
        try:
            new_parent = os.path.join(parentpath, path)
            paths += get_filepaths(new_parent, os.listdir(new_parent))
        except NotADirectoryError:
            paths.append(os.path.join(parentpath, path))
    return paths

def aesthietic_loss_caculate(img_path, clip_dir=None, clip_model_name='ViT-B/16', aesthetic_dir=None, device=None):
    clip_dir = clip_dir or os.getenv('AESTHETIC_CLIP_PATH')
    aesthetic_dir = aesthetic_dir or os.getenv('AESTHETIC_HEAD_PATH')
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not clip_dir or not aesthetic_dir:
        raise RuntimeError('请设置 AESTHETIC_CLIP_PATH 和 AESTHETIC_HEAD_PATH')
    model_dir = os.path.dirname(clip_dir)
    clip_model= clip.load(clip_model_name, jit=False, device=device, download_root=model_dir)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
                                     # 加载美学评分模型
    model = AestheticMeanPredictionLinearModel(512)
    model.load_state_dict(
        torch.load(aesthetic_dir)
    )
    model = model.to(device)
    img = Image.open(img_path).convert('RGB')
    img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (224, 224))
    img = TF.to_tensor(img).to(device)
    img = normalize(img)
    clip_image_embed = F.normalize(
    clip_model.encode_image(img[None, ...]).float(),dim=-1)
    score = model(clip_image_embed).item()
    return score
if __name__ == "__main__":
    print('请在调用 aesthietic_loss_caculate 时传入图片和模型路径')
