import torch
import os 
from .utils.bbox_visualization import bbox_visualization,scale_boxes
from PIL import Image
from .src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from .src.pipeline.pipeline_CreatiLayout_recontraster import CreatiLayoutSD3Pipeline
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from diffusers.schedulers import FlowMatchHeunDiscreteScheduler,DEISMultistepScheduler

def load_mask(file_path,device,target_size=(128,128)):
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

def detect_boundary(mask,device):
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

def boundary_get(mask_image_path, target_size=(128, 128)):
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(mask_image, 100, 200)
    edges_image = Image.fromarray(edges)
    transform = transforms.Compose([
        transforms.Resize( target_size),
        transforms.ToTensor()
    ])
    edges_tensor = transform(edges_image)
    return edges_tensor

def generate(inputs, output, model_path=None, ckpt_path=None, c_T=0,
             guidance_scale=3.5, scale_factor=1, seed=42,
             num_inference_steps=50, change_solver=False):
    model_path = model_path or os.getenv("SD3_MODEL_PATH")
    ckpt_path = ckpt_path or os.getenv("CREATILAYOUT_MODEL_PATH")
    if not model_path or not ckpt_path:
        raise RuntimeError(
            "请设置 SD3_MODEL_PATH 和 CREATILAYOUT_MODEL_PATH，"
            "或在调用 generate 时显式传入模型路径"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformer_additional_kwargs = dict(attention_type="layout",strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
         ckpt_path, subfolder="transformer", torch_dtype=torch.float16,**transformer_additional_kwargs)
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    if change_solver:
        print("Using DEISMultistepScheduler as the scheduler.")
        pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(pipe.scheduler.config)
    mask=load_mask(inputs["mask"],device)
    mask_boundary=detect_boundary(mask,device)
    boundary_line=boundary_get(inputs["mask"]).to(device)
    
    batch_size = 2
    height = 1024
    width = 1024

    prompts=[inputs["region1"],inputs["region2"]]
    boxes=inputs["box"]
    region_caption1=[]
    region_caption2=[]
    region_bboxes1=[]
    region_bboxes2=[]
    for key,value in boxes["region1"].items():
        region_caption1.append(key)
        region_bboxes1.append(value)
    for key,value in boxes["region2"].items():
        region_caption2.append(key)
        region_bboxes2.append(value)
    region_caption=[region_caption1,region_caption2]
    region_bboxes=[region_bboxes1,region_bboxes2]
    with torch.no_grad():
        images = pipe(prompt = prompts,
                generator = torch.Generator(device="cuda").manual_seed(seed),
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                bbox_phrases = region_caption, 
                bbox_raw = region_bboxes,
                height = height,
                width = width,
                region_mask=mask,
                region_boundary=mask_boundary,
                boundary_line=boundary_line,
                scale_factor=scale_factor,
                c_T=c_T
            )
    images=images.images
    images[0].save(output)

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
    mask=load_mask(mask_path)
    mask_boundary=detect_boundary(mask)

    seed = 42
    batch_size = 2
    num_inference_steps = 50
    guidance_scale = 3.5
    height = 1024
    width = 1024

    prompts=[prompts["region1"],prompts["region2"]]
    region_caption1=[]
    region_caption2=[]
    region_bboxes1=[]
    region_bboxes2=[]
    for key,value in boxes["region1"]:
        region_caption1.append(key)
        region_bboxes1.append(value)
    for key,value in boxes["region2"]:
        region_caption2.append(key)
        region_bboxes2.append(value)
    region_caption=[region_caption1,region_caption2]
    region_bboxes=[region_bboxes1,region_bboxes2]
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
                c_T=5
            )
    images=images.images
    images[0].save(out_path)
    # global_caption = [
    # "A young girl is standing confidently at the center of the image, smiling warmly. She is the main focus of the scene. The background features soft greenery and sunlight, but it is subtle to emphasize the girl's portrait.",
    # "A young girl is standing at the center of the image in a war-torn area. She looks distressed and is crying, symbolizing the horrors of conflict. The background shows ruins of destroyed buildings and smoke rising in the air, creating a grim and chaotic atmosphere.",
    # "A striking image shows a split-faced young girl. The left side of her face is smiling happily, symbolizing peace and innocence, while the right side of her face is crying in distress, symbolizing the horrors of war. The left background features a serene park with green grass, flowers, and soft sunlight. The right background shows a war-torn area with destroyed buildings, rubble, and rising smoke. The image strongly contrasts peace and war, innocence and suffering."
    # ]
    
    # region_caption_list = [[
    #     "A young girl smiling warmly, standing at the center of the image.",
    #     "Soft greenery in the blurred background.",
    #     "Gentle sunlight providing a warm glow behind the girl."
    # ],
    # [
    #     "A young girl crying, standing at the center of the image.",
    #     "Destroyed buildings and rubble in the background.",
    #     "Smoke rising into the sky, symbolizing destruction."
    # ],
    # [
    #     "A split-faced young girl. The left side of her face is smiling happily, representing peace and innocence, while the right side is crying in distress, representing the horrors of war.",
    #     "The left background is a peaceful park, while the right background is a war-torn area."
    # ]
    # ]
    
    # region_bboxes_list = [[
    #     [0.30, 0.20, 0.70, 0.80],  
    #     [0.00, 0.00, 1.00, 0.50],
    #     [0.00, 0.00, 1.00, 0.30]
    # ],
    # [
    #     [0.30, 0.20, 0.70, 0.80],  
    #     [0.00, 0.40, 1.00, 1.00],  
    #     [0.00, 0.00, 1.00, 0.40]
    # ],
    # [
    #     [0.30, 0.20, 0.70, 0.80],  
    #     [0.00, 0.00, 1.00, 1.00] 
    # ]
    # ]
    # global_caption = [
    # "A serene ocean under a bright blue sky, with a perfectly round, smooth sandy dune rising gently from the clear water's surface. The horizon is peaceful and expansive, with soft clouds and clear, reflective water.",
    # "A massive, sharply visible human skull eerily floating in a murky, heavily clouded underwater environment. The skull is perfectly aligned to face directly toward the viewer, with its hollow eye sockets staring menacingly. The water around the skull is gray and opaque, filled with swirling debris and fine particles, creating an unsettling, suffocating atmosphere.",
    # ]
    
    # region_caption_list = [
    #     [
    # "A perfectly round, smooth sandy dune with a single plant at its peak, rising gently from the water.",
    #     ],
    #     [
    #         "A massive, sharply visible human skull facing directly toward the viewer, surrounded by murky gray water filled with debris and particles.",
    #     ]

    # ]
    
    global_caption = [
        "A serene ocean under a bright blue sky, with a perfectly round, smooth sandy dune rising gently from the clear water's surface. The horizon is peaceful , with soft clouds and clear, reflective water.A tree with sparse foliage growing on the round, smooth sandy dune",
     "A massive human skull floats eerily in black blue-green sea. The skull stares menacingly with hollow sockets as the water fades into obscurity, creating a haunting atmosphere."
    ]
    
    region_caption_list = [
        [
    "A perfectly round, smooth sandy dune with a single plant at its peak, rising gently from the water.A tree with sparse foliage growing on the round, smooth sandy dune",
        ],
     [
    "A massive human skull facing directly toward the viewer, surrounded by thick, black blue-green waterfading into shadow, filled with swirling debris and fine particles."
    ]

    ]
    
    region_bboxes_list = [
    [
        [0.30, 0.20, 0.70, 0.30],  # 沙丘和植被区域，位于中央偏下
    ],
    [
        [0.20, 0.20, 0.80, 0.90],  # 骷髅正对视角，占据图像主要位置
    ]]

    filename = "Spider Man"

    with torch.no_grad():
        for i in range(50):
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
                        c_T=i
                    )
            images=images.images
            images[0].save(f'./img_show/{i}.png')
    
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

    
    
