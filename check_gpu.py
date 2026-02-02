#!/usr/bin/env python3
"""诊断 GPU 状态并尝试清理"""

import subprocess
import sys
import time

def run_cmd(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

print("=" * 80)
print("GPU 诊断工具")
print("=" * 80)

# 1. 检查 nvidia-smi
print("\n1. nvidia-smi 状态:")
print(run_cmd("nvidia-smi"))

# 2. 检查所有使用 GPU 的进程
print("\n2. 使用 GPU 的进程:")
print(run_cmd("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"))

# 3. 检查所有 Python 进程
print("\n3. 所有 Python 进程:")
print(run_cmd("ps aux | grep python"))

# 4. 尝试用 PyTorch 访问 GPU
print("\n4. PyTorch CUDA 测试:")
try:
    import torch
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   设备名称: {torch.cuda.get_device_name(0)}")
        
        # 尝试创建张量
        try:
            x = torch.zeros(1).cuda()
            print(f"   ✓ 成功创建 CUDA 张量: {x}")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ✗ 创建 CUDA 张量失败: {e}")
except ImportError:
    print("   ✗ PyTorch 未安装")
except Exception as e:
    print(f"   ✗ 错误: {e}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

# 5. 提供清理建议
print("\n如果发现问题，尝试以下命令:")
print("  # 杀掉所有 Python 进程:")
print("  pkill -9 python")
print("\n  # 或者杀掉特定 PID:")
print("  kill -9 <PID>")
print("\n  # 重置 GPU (需要 sudo):")
print("  sudo nvidia-smi --gpu-reset -i 0")
