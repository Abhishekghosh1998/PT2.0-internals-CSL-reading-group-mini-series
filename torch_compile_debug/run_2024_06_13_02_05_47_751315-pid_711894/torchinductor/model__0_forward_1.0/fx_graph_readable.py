class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1000]"):
        # File: /tmp/ipykernel_711894/2967604626.py:3 in f, code: return torch.sin(x)**2 + torch.cos(x)**2
        sin: "f32[1000]" = torch.ops.aten.sin.default(primals_1)
        pow_1: "f32[1000]" = torch.ops.aten.pow.Tensor_Scalar(sin, 2);  sin = None
        cos: "f32[1000]" = torch.ops.aten.cos.default(primals_1)
        pow_2: "f32[1000]" = torch.ops.aten.pow.Tensor_Scalar(cos, 2);  cos = None
        add: "f32[1000]" = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
        return [add, primals_1]
        