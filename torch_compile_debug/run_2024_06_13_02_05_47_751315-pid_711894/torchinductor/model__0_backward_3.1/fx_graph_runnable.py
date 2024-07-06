
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

    
    
    def forward(self, primals_1, tangents_1):
        cos = torch.ops.aten.cos.default(primals_1)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(cos, 1.0)
        mul = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_1 = torch.ops.aten.mul.Tensor(tangents_1, mul);  mul = None
        sin = torch.ops.aten.sin.default(primals_1);  primals_1 = None
        neg = torch.ops.aten.neg.default(sin)
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, neg);  mul_1 = neg = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sin, 1.0);  sin = None
        mul_3 = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_4 = torch.ops.aten.mul.Tensor(tangents_1, mul_3);  tangents_1 = mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, cos);  mul_4 = cos = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, mul_5);  mul_2 = mul_5 = None
        return [add_1]
        
def load_args(reader):
    buf0 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1000,), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1000,), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)