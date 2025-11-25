#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform

print("Python version:", platform.python_version())
print("Python full info:", platform.python_version(), platform.python_build(), platform.python_compiler())

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
except ImportError:
    print("PyTorch is not installed.")
