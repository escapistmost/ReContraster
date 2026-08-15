import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import torch


def boundary_get(mask_image_path, target_size=(1024, 1024)):
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.resize(mask_image, target_size)
    edges = cv2.Canny(mask_image, 100, 200)
    edges_image = Image.fromarray(edges)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    edges_tensor = transform(edges_image)
    return edges_tensor


def mask_get(mask_image_path, target_size=(1024, 1024)):
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.resize(mask_image, target_size)
    mask_image = Image.fromarray(mask_image)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mask_tensor = transform(mask_image)
    return mask_tensor.squeeze()


def extract_grad(tensor):
    # 提取图像的x、y方向梯度
    """
    使用Sobel算子计算tensor的梯度
    Args:
        tensor (torch.Tensor): 输入的tensor变量，大小为[ H, W]。

    Returns:
        (torch.Tensor, torch.Tensor): 返回水平（dx）和垂直（dy）方向的梯度
    """
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]]).float().unsqueeze(0).unsqueeze(0).to(tensor.dtype)

    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 1.0]]).float().unsqueeze(0).unsqueeze(0).to(tensor.dtype)
    # sobel_x = torch.tensor([[1, 2 ,0,-2, -1],
    #                         [4,8,0,-8,-4],
    #                         [6,12,0,-12,-6],                                                                     
    #                         [4,8,0,-8,-4],
    #                         [1,2,0,-2,-1]]).float().unsqueeze(0).unsqueeze(0).to(tensor.dtype)

    # sobel_y = torch.tensor([[1, 4 ,6,4, 1],
    #                         [2,8,12,8,2],
    #                         [0,0,0,0,0],                                                                     
    #                         [-2,-8,-12,-8,-2],
    #                         [-1, -4 ,-6,-4, -1]]).float().unsqueeze(0).unsqueeze(0).to(tensor.dtype)

    # 扩展成4D tensor以适配卷积
    sobel_x = sobel_x.to(tensor.device)
    sobel_y = sobel_y.to(tensor.device)

    tensor = tensor.unsqueeze(0).unsqueeze(0)  # 从 [H, W] -> [1, 1, H, W]
    # 使用卷积计算梯度
    # grad_x = F.conv2d(tensor, sobel_x, padding='same')
    # grad_y = F.conv2d(tensor, sobel_y, padding='same')

    grad_x = F.conv2d(tensor, sobel_x, padding=1, groups=tensor.size(1))
    grad_y = F.conv2d(tensor, sobel_y, padding=1, groups=tensor.size(1))

    return grad_x.squeeze(0), grad_y.squeeze(0)


def grad_norm(gradx, grady):
    # 梯度归一化，防止过大过小
    xmax = gradx.max()
    xmin = gradx.min()
    ymax = grady.max()
    ymin = grady.min()
    if (xmax - xmin) > (ymax - ymin):
        value_g = xmax - xmin
    else:
        value_g = ymax - ymin
    return gradx / value_g, grady / value_g


def caculate_loss(grad1_x, grad2_x, grad1_y, grad2_y, boundary, eps=1e-6):
    # 依据梯度和boundary计算损失
    grad1_x, grad1_y = grad_norm(grad1_x, grad1_y)
    grad2_x, grad2_y = grad_norm(grad2_x, grad2_y)
    dot = grad1_x * grad2_x + grad1_y * grad2_y

    norm1 = torch.sqrt(grad1_x ** 2 + grad1_y ** 2 + eps)
    norm2 = torch.sqrt(grad2_x ** 2 + grad2_y ** 2 + eps)
    # cosine = dot / (norm1 * norm2 + eps)  # 余弦相似度

    # 添加数值稳定性检查
    denominator = norm1 * norm2 + eps
    cosine = dot / torch.clamp(denominator, min=eps)  # 确保分母不小于eps

    cosine = cosine ** 2
    # boundary=boundary.to('cuda')
    boundary_cosine = cosine * boundary
    num_boundary_pixels = boundary.sum() + eps
    mean_cosine = boundary_cosine.sum() / num_boundary_pixels
    loss = 1 - mean_cosine
    return loss


def update_latent(latent: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
    # 依据损失修改latent
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latent], retain_graph=True)[0]
    latent = latent - step_size * grad_cond
    return latent


def testforloss(latent, edges_tensor):
    grad_x = []
    grad_y = []
    gx, gy = extract_grad(latent)
    grad_x.append(gx)
    grad_y.append(gy)
    loss = caculate_loss(grad_x[0], grad_y[0], grad_x[1], grad_y[1], edges_tensor)
    return loss


def get_img(imgpath='0.png'):
    # 读取彩色图像
    image_path = imgpath
    image = cv2.imread(image_path)
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将灰度图像转换为Tensor
    gray_image_float = gray_image.astype(np.float32) / 255.0
    tensor_image = torch.tensor(gray_image_float)
    # tensor_image = tensor_image.unsqueeze(0)  # [1, H, W]
    # print(f"Tensor shape: {tensor_image.shape}")
    return tensor_image

def renew_grad(x,y,mask_x,mask_y,boundary, eps=1e-6):
    # 修正梯度，去除mask的影响（因为我们计算某一侧梯度是要把另一边删掉变成黑色，因此要去一个均值，减少黑色区域与原图产生的梯度）
    norm=torch.sqrt(mask_x ** 2 + mask_y ** 2 + eps)

    boundary_sum = boundary.sum()
    if boundary_sum < 1e-6:
        return x, y  # 如果边界区域为空，直接返回原始梯度
    
    sin=mask_y/norm
    cos=mask_x/norm
    avg_x=x.sum()/boundary.sum()
    avg_y=y.sum()/boundary.sum()
    new_x=x-avg_x*cos
    new_y=y-avg_y*sin
    return new_x,new_y


def boundary_loss_caculate(img_path, mask_path):
    # 计算损失
    img = get_img(img_path)
    mask_boundary = boundary_get(mask_path, img.shape)
    mask = mask_get(mask_path, img.shape)
    img1 = img * mask
    img2 = img * (1 - mask)
    
    mask_grad_x, mask_grad_y = extract_grad(mask)
    mask_grad_x = mask_grad_x * mask_boundary
    mask_grad_y = mask_grad_y * mask_boundary
    mask_grad_x, mask_grad_y = grad_norm(-mask_grad_x, -mask_grad_y)

    img1_x, img1_y = extract_grad(img1)
    img2_x, img2_y = extract_grad(img2)

    img1_x = img1_x * mask_boundary
    img1_y = img1_y * mask_boundary
    img1_x, img1_y = renew_grad(img1_x, img1_y, mask_grad_x, mask_grad_y, mask_boundary)
    img1_x, img1_y = grad_norm(img1_x, img1_y)

    img2_x = img2_x * mask_boundary
    img2_y = img2_y * mask_boundary
    img2_x, img2_y = renew_grad(img2_x, img2_y, mask_grad_x, mask_grad_y, mask_boundary)
    img2_x, img2_y = grad_norm(img2_x, img2_y)

    loss = caculate_loss(img1_x, img2_x, img1_y, img2_y, mask_boundary)
    return loss

if __name__ == '__main__':
    print('请在调用 boundary_loss_caculate 时传入图片和 mask 路径')
