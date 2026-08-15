import cv2
import matplotlib.colors as mcolors
import colorsys
from sklearn.cluster import KMeans

def hex_to_rgb(hex_color):
    # hex_color = "#FF5733"  # 颜色代码
    rgb_color = mcolors.hex2color(hex_color)  # 归一化的RGB (0-1)
    rgb_color_255 = tuple(int(c * 255) for c in rgb_color)  # 转换为 0-255 范围
    return rgb_color_255

def rgb_to_hex(rgb_color):
    # 确保 RGB 值在 0-255 范围内
    return '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])

def resize_mask_to_image(mask, image_shape):
    # 将mask调整为和图像相同的尺寸
    return cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

def extract_colored_regions(image, mask, region_value):
    # 提取mask中指定区域的颜色（黑色或白色区域）
    if region_value == 255:
        # 白色区域
        region_pixels = image[mask == 255]
    else:
        # 黑色区域
        region_pixels = image[mask == 0]
    return region_pixels

def get_top_k_colors(image_path, mask_path, k=10):
    # 读取图像和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # 读取灰度图

    # 将mask调整为和图像大小一致
    mask_resized = resize_mask_to_image(mask, image.shape)

    # 提取白色区域的颜色
    white_region_pixels = extract_colored_regions(image, mask_resized, region_value=255)
    # 提取黑色区域的颜色
    black_region_pixels = extract_colored_regions(image, mask_resized, region_value=0)

    # 如果没有颜色区域，直接返回空
    if white_region_pixels.size == 0 and black_region_pixels.size == 0:
        return [], []

    # 使用K-means聚类来分析白色区域的颜色
    kmeans_white = KMeans(n_clusters=k)
    if white_region_pixels.size > 0:
        kmeans_white.fit(white_region_pixels)
        white_cluster_centers = kmeans_white.cluster_centers_.astype(int)
    else:
        white_cluster_centers = []

    # 使用K-means聚类来分析黑色区域的颜色
    kmeans_black = KMeans(n_clusters=k)
    if black_region_pixels.size > 0:
        kmeans_black.fit(black_region_pixels)
        black_cluster_centers = kmeans_black.cluster_centers_.astype(int)
    else:
        black_cluster_centers = []

    return white_cluster_centers, black_cluster_centers

def plot_colors(colors, title):
    # 创建一个小方格显示每个提取的颜色
    n_colors = len(colors)
    square_size = 50  # 每个小方格的大小

    # 创建一个空的画布
    color_canvas = np.zeros((square_size, square_size * n_colors, 3), dtype=int)

    # 绘制每个小方格
    for i, color in enumerate(colors):
        color_canvas[:, i * square_size: (i + 1) * square_size, :] = color

    # 显示颜色方格
    plt.figure(figsize=(n_colors, 2))
    plt.imshow(color_canvas)
    plt.axis('off')
    plt.title(title)
    plt.show()


def get_opposite_color(rgb):
    # 将RGB颜色转换为HSV
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # 获取对侧颜色的色相（+180度）
    opposite_hue = (h + 0.5) % 1.0  # 在0到1之间，增加0.5即相差180度

    # 获取对侧颜色的饱和度和明度
    opposite_saturation = (s + 0.6) % 1.0  # 取反饱和度
    opposite_value = (v + 0.6) % 1.0 # 取反明度
    # opposite_saturation = 1 - s  # 取反饱和度
    # opposite_value = 1 - v  # 取反明度
    # 将对侧颜色的HSV转换回RGB
    r_opposite, g_opposite, b_opposite = colorsys.hsv_to_rgb(opposite_hue, opposite_saturation, opposite_value)
    return rgb_to_hex((int(r_opposite * 255), int(g_opposite * 255), int(b_opposite * 255)))

def list_opposite(color_list):
    color_new=[]
    for i in color_list:
        color_new.append(get_opposite_color(i))
    return  color_new

def get_opposite(img,mask,k=5):
    white_colors, black_colors = get_top_k_colors(img, mask, k)
    white_colors = list_opposite(white_colors)
    black_colors = list_opposite(black_colors)
    return  [white_colors,black_colors]