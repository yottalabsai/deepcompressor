# Enhanced Nunchaku Convert with Config.json Metadata Support

## 概述

这个增强功能为 `deepcompressor` 的 nunchaku 后端添加了将 `config.json` 文件中的元数据信息嵌入到生成的 `safetensors` 文件中的能力。该功能特别适用于 Flux 等扩散模型的量化转换。

## 功能特性

- ✅ 自动从模型的 `transformer/config.json` 文件读取配置信息
- ✅ 将配置信息作为元数据嵌入到 `safetensors` 文件中
- ✅ 支持从扩散模型配置文件（如 `flux.1-dev.yaml`）自动检测模型路径
- ✅ 提供独立的后处理工具来增强现有的 `safetensors` 文件
- ✅ 包含元数据验证和检查工具
- ✅ 向后兼容原有的转换功能

## 新增文件

- `convert_enhanced.py` - 核心元数据处理功能
- `validate_metadata.py` - 元数据验证工具
- `README_metadata.md` - 本文档

## 使用方法

### 1. 增强的转换脚本

使用增强的 convert 脚本，添加 `--add-metadata` 参数：

```bash
python -m deepcompressor.backend.nunchaku.convert \
    --quant-path /path/to/quantization/checkpoint \
    --output-root /path/to/output \
    --model-name flux.1-dev \
    --add-metadata \
    --config-yaml examples/diffusion/configs/model/flux.1-dev.yaml \
    --model-path /path/to/flux.1-dev/model
```

### 2. 独立的元数据增强工具

对现有的 `safetensors` 文件添加元数据：

```bash
python -m deepcompressor.backend.nunchaku.convert_enhanced \
    --safetensors-path /path/to/transformer_blocks.safetensors \
    --config-yaml examples/diffusion/configs/model/flux.1-dev.yaml \
    --model-path /path/to/flux.1-dev/model
```

### 3. 元数据验证

验证 `safetensors` 文件中的元数据：

```bash
# 基本验证
python -m deepcompressor.backend.nunchaku.validate_metadata \
    /path/to/transformer_blocks.safetensors

# 详细验证（显示元数据内容）
python -m deepcompressor.backend.nunchaku.validate_metadata \
    /path/to/transformer_blocks.safetensors --verbose

# 比较两个文件的元数据
python -m deepcompressor.backend.nunchaku.validate_metadata \
    /path/to/file1.safetensors --compare /path/to/file2.safetensors
```

## 程序化使用

```python
from deepcompressor.backend.nunchaku.convert_enhanced import (
    load_config_json, 
    prepare_metadata_from_config,
    enhance_safetensors_with_config_metadata
)
import safetensors.torch

# 方法 1：在保存时添加元数据
model_path = "/path/to/flux.1-dev/model"
config_data = load_config_json(model_path)
metadata = prepare_metadata_from_config(config_data)

safetensors.torch.save_file(
    tensors_dict, 
    "output.safetensors", 
    metadata=metadata
)

# 方法 2：增强现有文件
success = enhance_safetensors_with_config_metadata(
    safetensors_path="/path/to/transformer_blocks.safetensors",
    config_yaml_path="examples/diffusion/configs/model/flux.1-dev.yaml"
)
```

## 元数据结构

添加到 `safetensors` 文件的元数据包括：

```json
{
  "model_type": "flux",
  "architectures": "[\"FluxTransformer2DModel\"]",
  "hidden_size": "3072",
  "num_hidden_layers": "19", 
  "num_attention_heads": "24",
  "intermediate_size": "12288",
  "max_position_embeddings": "512",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.0",
  "full_config": "{...完整的config.json内容...}",
  "metadata_source": "config.json",
  "enhanced_by": "deepcompressor_nunchaku_convert_enhanced"
}
```

## 路径检测逻辑

1. **明确指定路径**: 使用 `--model-path` 参数直接指定模型目录
2. **从配置文件检测**: 从 `--config-yaml` 指定的文件中提取 `pipeline.path` 字段
3. **使用模型名称**: 如果没有找到路径，使用模型名称作为默认路径

## config.json 搜索顺序

1. `{model_path}/transformer/config.json`
2. `{model_path}/config.json`
3. `{model_path}/transformer.config.json`

## 错误处理

- 如果找不到 `config.json` 文件，会显示警告但继续执行转换
- 如果元数据处理失败，会显示警告并保存不带元数据的文件
- 自动创建备份文件（`.backup`）以防止数据丢失

## 向后兼容性

- 不使用 `--add-metadata` 参数时，功能与原版完全相同
- 现有的调用方式不受影响
- 原有的 API 接口保持不变

## 示例和测试

参考 `examples/nunchaku_metadata_example.py` 文件获取完整的使用示例和最佳实践。

## 依赖项

- `safetensors` - 用于处理 safetensors 文件格式
- `PyYAML` - 用于解析配置文件
- `json` - 用于处理 JSON 数据

## 注意事项

1. **元数据限制**: safetensors 的元数据只支持字符串值，所有非字符串值会被转换为字符串
2. **文件大小**: 添加元数据会略微增加文件大小，但影响很小
3. **备份**: 增强现有文件时会自动创建备份（`.backup`）
4. **权限**: 确保对目标文件和目录有写权限

## 故障排除

### 常见问题

1. **找不到 config.json**
   - 检查模型路径是否正确
   - 确认 config.json 文件确实存在于 transformer 目录下
   - 尝试使用 `--model-path` 明确指定路径

2. **元数据验证失败**
   - 使用 `validate_metadata.py` 检查具体错误
   - 确认 safetensors 文件没有损坏

3. **权限错误**
   - 确保对输出目录有写权限
   - 检查原文件是否被其他进程占用

### 调试

启用详细输出来诊断问题：

```bash
python -m deepcompressor.backend.nunchaku.convert \
    --add-metadata \
    --config-yaml examples/diffusion/configs/model/flux.1-dev.yaml \
    # ... 其他参数
```

该功能会在处理过程中显示详细的状态信息。
