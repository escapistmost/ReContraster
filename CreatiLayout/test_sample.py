import torch
import os 
from utils.bbox_visualization import bbox_visualization,scale_boxes
from PIL import Image
from src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from src.pipeline.pipeline_CreatiLayout_recontraster import CreatiLayoutSD3Pipeline
import numpy as np
import torch.nn.functional as F
def load_mask(file_path,target_size=(128,128)):
    """
    加载 mask 图像文件，并将其转换为指定尺寸的二值化 Tensor
    """
    # 打开图像并转换为灰度
    mask_image = Image.open(file_path).convert('L')
    
    # 调整大小到 target_size
    mask_image = mask_image.resize(target_size, Image.NEAREST)
    
    # 转换为 numpy 数组
    mask_array = np.array(mask_image)
    
    # 二值化处理（假设原始 mask 是 0 和 255）
    mask_array = (mask_array > 128).astype(np.float16)  # 将值转换为 0 或 1
    
    # 转换为 PyTorch Tensor
    mask_tensor = torch.tensor(mask_array, dtype=torch.float16).to(device)
    
    return mask_tensor

def detect_boundary(mask):
    """
    检测 mask 中的边界，返回边界区域的 mask
    """
    # 卷积核用来计算 mask 的梯度，用来找到边界
    kernel = torch.tensor([[1, -1]], dtype=torch.float16).view(1, 1, 1, 2).to(device)  # 用于水平梯度
    mask_4d = mask.view(1, 1, *mask.shape).to(device)   # 将 mask 转为 4D (batch, channel, height, width)

    # 计算水平和垂直方向的边界，使用 'same' padding 保持维度一致
    grad_x = F.conv2d(mask_4d, kernel, padding=(0, 1))[:, :, :, :-1]  # 水平方向边界，去掉多余的列
    grad_y = F.conv2d(mask_4d, kernel.transpose(2, 3), padding=(1, 0))[:, :, :-1, :]  # 垂直方向边界，去掉多余的行

    # 边界 mask：只要水平或垂直方向有梯度，就是边界
    boundary_mask = (torch.abs(grad_x) > 0) | (torch.abs(grad_y) > 0)
    boundary_mask = boundary_mask.view(*mask.shape).float().to(device)  # 转为 2D
    return boundary_mask

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.getenv("SD3_MODEL_PATH")
    ckpt_path = os.getenv("CREATILAYOUT_MODEL_PATH")
    if not model_path or not ckpt_path:
        raise RuntimeError("请设置 SD3_MODEL_PATH 和 CREATILAYOUT_MODEL_PATH")
    transformer_additional_kwargs = dict(attention_type="layout",strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
         ckpt_path, subfolder="transformer", torch_dtype=torch.float16,**transformer_additional_kwargs)
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    mask=load_mask('./input/mask.png')
    mask_boundary=detect_boundary(mask)

    seed = 42
    batch_size = 2
    num_inference_steps = 30
    guidance_scale = 7.5
    height = 1024
    width = 1024

    save_root = "output"
    img_save_root = os.path.join(save_root,"images")
    os.makedirs(img_save_root,exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root,"images_with_layout")
    os.makedirs(img_with_layout_save_root,exist_ok=True)
    global_caption = [
    "A serene ocean under a bright blue sky, with a perfectly round, smooth sandy dune rising gently from the clear water's surface. The horizon is peaceful and expansive, with soft clouds and clear, reflective water.",
    "A massive, sharply visible human skull eerily floating in a deep blue, heavily clouded underwater environment. The skull is perfectly aligned to face directly toward the viewer, with its hollow eye sockets staring menacingly. The water is dark but not black, filled with swirling debris and fine particles that enhance the murky atmosphere.",
    # "A surreal, cold portrait of a woman with pale skin, sharp features, and a detached expression. Her frozen hair intertwines with jagged glass and metal. The dark, desolate background is filled with geometric shapes and cold symbols. Her clothing is minimal and angular, with metallic textures"
    ]
    
    region_caption_list = [
        [
    "A perfectly round, smooth sandy dune with a single plant at its peak, rising gently from the water.",
    # "Calm, clear water gently rippling around the dune, creating a seamless transition to the underwater area.",
    # "A bright blue sky with soft, scattered clouds drifting peacefully above the horizon."
        ],
        [
            "A massive, sharply visible human skull facing directly toward the viewer.",
            "Deep blue, heavily clouded water filled with swirling debris and fine particles surrounding the skull.",
            "Soft gradients of cold blue light emanating from the skull, contrasting with the darker, murky water."
        ]
        # [ 
        #     "A surreal, hyper-realistic portrait of a woman with cold, detached expression, front-facing.",
        #     "Her pale, translucent skin contrasts with the dark, desolate background.",
        #     "Her hair appears frozen and brittle, intertwining with jagged, cold elements like glass or metal.",
        #     "Floating geometric shapes or cold symbols surround her, adding to the isolated atmosphere.",
        #     "Her clothing is minimal, angular, with sharp, metallic textures and geometric patterns."
        # ]
    ]
    
    region_bboxes_list = [
    [
        [0.30, 0.25, 0.70, 0.40],  # 沙丘和植被区域，位于中央偏下
        # [0.00, 0.30, 1.00, 0.35],  # 清澈的水流和海面过渡区域
        # [0.00, 0.00, 1.00, 0.25],  # 天空和云层区域，占据上半部分
    ],
    [
        [0.20, 0.20, 0.80, 0.90],  # 骷髅正对视角，占据图像主要位置
        [0.00, 0.10, 1.00, 0.90],  # 水域漂浮物和浑浊区域，覆盖大部分背景
        [0.00, 0.70, 1.00, 1.00],  # 下方冷蓝色渐变光区域
    ]]



    filename = "Spider Man"

    with torch.no_grad():
        images = pipe(prompt = global_caption,
                    generator = torch.Generator(device="cuda").manual_seed(seed),
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    bbox_phrases = region_caption_list, 
                    bbox_raw = region_bboxes_list,
                    height = height,
                    width = width,
                    region_mask=mask,
                    region_boundary=mask_boundary,
                    c_T=50
                )
        images=images.images
    
    for j, image in enumerate(images):   

        image.save(os.path.join(img_save_root,f"{filename}_{j}.png")) 

        img_with_layout_save_name=os.path.join(img_with_layout_save_root,f"{filename}_{j}.png")

        white_image = Image.new('RGB', (width, height), color='rgb(256,256,256)')
        show_input = {"boxes":scale_boxes(region_bboxes_list[j],width,height),"labels":region_caption_list[j]}

        bbox_visualization_img = bbox_visualization(white_image,show_input)
        image_with_bbox = bbox_visualization(image ,show_input)

        total_width = width*2
        total_height = height

        new_image = Image.new('RGB', (total_width, total_height))
        new_image.paste(bbox_visualization_img, (0, 0))
        new_image.paste(image_with_bbox, (width, 0))
        new_image.save(img_with_layout_save_name)

    
    
