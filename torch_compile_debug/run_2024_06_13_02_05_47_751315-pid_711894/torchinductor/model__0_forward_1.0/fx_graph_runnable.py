
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.graph_diagram = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.4.0.dev20240530+cu121
# torch cuda version: 12.1
# torch git version: 0bbe39cc0e93abcde8ca1efa435c13bbe484b36b


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Wed_Nov_22_10:17:15_PST_2023 
# Cuda compilation tools, release 12.3, V12.3.107 
# Build cuda_12.3.r12.3/compiler.33567101_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 2080 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1):
        sin = torch.ops.aten.sin.default(primals_1)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sin, 2);  sin = None
        cos = torch.ops.aten.cos.default(primals_1)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(cos, 2);  cos = None
        add = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
        return [add, primals_1]
        
def load_args(reader):
    buf0 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1000,), is_leaf=True)  # primals_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)