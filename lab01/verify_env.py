# -*- coding: utf-8 -*-
"""环境验证脚本：确认 PyTorch 等依赖可用（对应实验指南 2.2 节步骤 4）。"""
import torch, sklearn, matplotlib
print("PyTorch:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())  # 无 GPU 显示 False 也可正常完成本实验
print("scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
