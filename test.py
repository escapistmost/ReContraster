import torch

print("=" * 50)
print("PyTorch CUDA 检测")
print("=" * 50)

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"\nCUDA 是否可用: {cuda_available}")

if cuda_available:
    # 获取CUDA设备数量
    device_count = torch.cuda.device_count()
    print(f"CUDA 设备数量: {device_count}")
    
    # 获取当前CUDA设备
    current_device = torch.cuda.current_device()
    print(f"当前CUDA设备ID: {current_device}")
    
    # 获取设备名称
    device_name = torch.cuda.get_device_name(current_device)
    print(f"设备名称: {device_name}")
    
    # 获取CUDA版本
    print(f"CUDA 版本: {torch.version.cuda}")
    
    # 获取cuDNN版本
    if torch.backends.cudnn.is_available():
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN 是否启用: {torch.backends.cudnn.enabled}")
    
    # 显示所有GPU信息
    print("\n所有GPU设备信息:")
    for i in range(device_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # 显示显存信息
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"    总显存: {total_memory:.2f} GB")
    
    # 测试创建一个tensor到GPU
    try:
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"\n✓ 成功创建GPU tensor: {test_tensor.device}")
    except Exception as e:
        print(f"\n✗ 创建GPU tensor失败: {e}")
else:
    print("\n⚠ CUDA不可用，将使用CPU进行计算")
    print("可能的原因:")
    print("  1. 没有安装NVIDIA GPU")
    print("  2. 没有安装CUDA toolkit")
    print("  3. 安装的PyTorch版本不支持CUDA")
    print("  4. GPU驱动问题")

# 显示PyTorch版本
print(f"\nPyTorch 版本: {torch.__version__}")

print("=" * 50)