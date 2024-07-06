class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1000]", tangents_1: "f32[1000]"):
        # File: /tmp/ipykernel_711894/2967604626.py:3 in f, code: return torch.sin(x)**2 + torch.cos(x)**2
        cos: "f32[1000]" = torch.ops.aten.cos.default(primals_1)
        pow_3: "f32[1000]" = torch.ops.aten.pow.Tensor_Scalar(cos, 1.0)
        mul: "f32[1000]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_1: "f32[1000]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  mul = None
        sin: "f32[1000]" = torch.ops.aten.sin.default(primals_1);  primals_1 = None
        neg: "f32[1000]" = torch.ops.aten.neg.default(sin)
        mul_2: "f32[1000]" = torch.ops.aten.mul.Tensor(mul_1, neg);  mul_1 = neg = None
        pow_4: "f32[1000]" = torch.ops.aten.pow.Tensor_Scalar(sin, 1.0);  sin = None
        mul_3: "f32[1000]" = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_4: "f32[1000]" = torch.ops.aten.mul.Tensor(tangents_1, mul_3);  tangents_1 = mul_3 = None
        mul_5: "f32[1000]" = torch.ops.aten.mul.Tensor(mul_4, cos);  mul_4 = cos = None
        
        # File: /tmp/ipykernel_711894/2967604626.py:3 in f, code: return torch.sin(x)**2 + torch.cos(x)**2
        add_1: "f32[1000]" = torch.ops.aten.add.Tensor(mul_2, mul_5);  mul_2 = mul_5 = None
        return [add_1]
        