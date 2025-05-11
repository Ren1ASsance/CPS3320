
import os

print("\n环境变量检查:")
print(f"  - CUDA_PATH: {os.environ.get('CUDA_PATH', '未设置')}")
print(f"  - PATH 中的 CUDA: {'cuda' in os.environ['PATH'].lower()}")