# 命名规范

## 核心原则

**公布的代码是全新整理过的，不是把现有代码原样搬过去。** 所有命名必须统一为 TriAttention 体系，内部开发名字不应出现在公布版本中。

## 命名映射表

| 类型 | 内部名字（不公布） | 公布名字 |
|------|-------------------|----------|
| 文件名 | `speckv_rkv_style.py` | `triattention.py` |
| 类名 | `SpeckVRKVStyle` | `TriAttention` |
| 函数名 | `apply_speckv_rkv_style_patch()` | `apply_triattention_patch()` |
| 配置key | `speckv_budget` | `triattention_budget` |
| 方法标识 | `method: speckv` | `method: triattention` |
| 脚本名 | `run_speckv_aime24_*.sh` | `run_triattention_*.sh` |
| 包名（baseline） | `rkv/compression/` | `kv_compress/` |
| 包名（我们的方法） | `rkv/compression/speckv.py` | `triattention/` |

## 重命名执行原则

1. **能小改就小改** — 能通过改变量名/函数名解决的，就小改，不要重构
2. **重构前必须报告** — 实在需要重构时，先说明：要改什么、为什么
3. **重构后必须验证** — 严格的 AB 对比测试 + 单元测试，确保行为一致

## 文件名中的 "aime" 泛化

脚本名中的 "aime" 需要泛化处理，例如：
- `run_speckv_aime24_*.sh` → `run_triattention_*.sh`
- 校准结果文件名中不能出现 "aime"
