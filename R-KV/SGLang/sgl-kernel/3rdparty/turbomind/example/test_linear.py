import torch
import torch.nn as nn
from safetensors import safe_open

import turbomind as tm
from turbomind.utils import unpack_awq_gemm

torch.manual_seed(0)


def i32x8_to_i4x8(w):
    """merge 8 integers (range from 0 to 15) into one 32-bit integer."""
    assert w.shape[-1] % 8 == 0
    shape = (w.shape[0], w.numel() // (w.shape[0] * 8), 8)
    shape = shape[:-1] + (1, )
    result = torch.zeros(shape, dtype=w.dtype, device=w.device)
    mask = torch.tensor([15], dtype=w.dtype, device=w.device)
    for i in range(8):
        shift = 4 * (7 - i)
        result[..., 0] |= (w[..., i] & mask) << shift
    result = result.view(w.shape[0], -1)
    return result


def makeup_weights(in_features: int, out_features: int, group_size: int = 128):
    # make up qweight
    assert out_features % 8 == 0
    qweight = torch.randint(0,
                            16, (in_features, out_features // 8, 8),
                            dtype=torch.int32,
                            device='cuda')
    print(f'-- makeup qweight: shape {qweight.shape}')
    print(qweight.view(in_features, -1))
    qweight = i32x8_to_i4x8(qweight)
    print(f'-- merge qweight: shape {qweight.shape}')
    print(qweight)

    # make up qzeros
    assert in_features % group_size == 0 and in_features // group_size >= 1
    qzeros = torch.randint(0,
                           16,
                           (in_features // group_size, out_features // 8, 8),
                           dtype=torch.int32,
                           device='cuda')
    print(f'-- makeup qzero: shape {qzeros.shape}')
    print(qzeros.view(in_features // group_size, -1))
    qzeros = i32x8_to_i4x8(qzeros)
    print(f'-- merge qzero: shape {qzeros.shape}\n{qzeros}')

    # make up scales
    scales = torch.rand((in_features // group_size, out_features),
                        dtype=torch.float16,
                        device='cuda')
    print(f'-- makeup scales: shape {scales.shape}\n{scales}')
    return qweight, qzeros, scales


def dequantize(qweight, qzeros, scales, group_size: int = 128):
    _qweight = unpack_awq_gemm(qweight)
    _qzeros = unpack_awq_gemm(qzeros)
    _qzeros = _qzeros.float()
    _qweight = _qweight.float()
    _scales = scales.float()
    for i in range(qzeros.shape[0]):
        start = i * group_size
        end = start + group_size
        _qweight[start:end] = (_qweight[start:end, :] -
                               _qzeros[i:i + 1, :]) * _scales[i:i + 1, :]
    return _qweight.half()


def load_specified_linear_weights():
    ckpt_path = '/models/140/llama3/Meta-Llama-3-8B-Instruct-hf-AWQ/model-00001-of-00002.safetensors'  # noqa
    layer_id = 0
    # prefix = f'model.layers.{layer_id}.self_attn.q_proj.'
    prefix = f'model.layers.{layer_id}.self_attn.o_proj.'
    keys = ['qweight', 'qzeros', 'scales']
    tensors = {}
    with safe_open(ckpt_path, framework='pt', device='cuda') as f:
        for key in keys:
            tensors[key] = f.get_tensor(prefix + key)

    return tensors['qweight'], tensors['qzeros'], tensors['scales']


# qweight, qzeros, scales = load_specified_linear_weights()
# in_features = qweight.shape[0]
# out_features = qweight.shape[1] * 8

group_size = 128
batch_size = 16384
in_features = 16384
out_features = 16384
qweight, qzeros, scales = makeup_weights(in_features, out_features, group_size)

x = torch.randn((batch_size, in_features),
                device=qweight.device,
                dtype=torch.float16)

weight = dequantize(qweight, qzeros, scales, group_size)
print(f'-- dequantization: weight.shape={weight.shape}, weight: \n{weight}')
ref_linear = nn.Linear(in_features, out_features, bias=False, device='cuda')
with torch.no_grad():
    ref_linear.weight = nn.Parameter(weight.T)
    ref_res = ref_linear(x)
    print(f'nn.linear.res: {ref_res}')

model = tm.Linear(in_features=in_features,
                  out_features=out_features,
                  bias=False,
                  quant_method='awq',
                  w_bit=4,
                  group_size=group_size)

model.qweight = qweight
model.qzeros = qzeros
model.scales = scales

model.post_init()

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    res = model(x)
stream.synchronize()

print(f'tm.linear.res: {res}')
abs_diff = torch.abs(res - ref_res).float()
rel_diff = abs_diff / torch.max(torch.abs(ref_res), torch.abs(res))
rtol = 0.01
atol = 0.0001
outliers = abs_diff > atol + rtol * torch.abs(ref_res)
abs_diff = torch.sum(abs_diff) / abs_diff.numel()
rel_diff = torch.sum(rel_diff) / rel_diff.numel()
outliers = torch.sum(outliers) / outliers.shape[0]
print(f'abs_diff {abs_diff:4f}, '
      f'rel_diff {rel_diff:4f}, '
      f'outliers {outliers:4f}')
