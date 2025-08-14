# Hugging Face 到 ModelScope 模型迁移脚本总结

## 📋 提供的脚本概览

我为您创建了一套完整的模型迁移工具，包含多个脚本以适应不同的使用场景。

### 🛠️ 核心脚本

| 脚本 | 功能 | 复杂度 | 适用场景 |
|-----|------|--------|----------|
| **`hf_to_modelscope_transfer.py`** | 完整功能迁移脚本 | 高 | 生产环境，需要详细控制 |
| **`simple_hf_to_ms.py`** | 简化迁移脚本 | 低 | 快速迁移，简单使用 |
| **`batch_transfer.sh`** | 批量处理脚本 | 中 | 多模型批量迁移 |
| **`quick_start_transfer.sh`** | 快速开始脚本 | 低 | 新手入门，环境设置 |

### 🎯 辅助文件

| 文件 | 用途 |
|------|------|
| **`examples/model_transfer_list.txt`** | 示例模型列表 |
| **`docs/HF-to-ModelScope-Transfer.md`** | 详细使用指南 |

## 🚀 快速开始

### 1. 环境设置
```bash
# 一键设置环境
bash scripts/quick_start_transfer.sh setup
```

### 2. 单个模型迁移
```bash
# 方式1: 使用快速脚本
bash scripts/quick_start_transfer.sh transfer microsoft/DialoGPT-small your-org/DialoGPT-small

# 方式2: 直接使用简化脚本
python scripts/simple_hf_to_ms.py microsoft/DialoGPT-small your-org/DialoGPT-small

# 方式3: 使用完整功能脚本
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "microsoft/DialoGPT-small" \
  --ms-model "your-org/DialoGPT-small"
```

### 3. 批量迁移
```bash
# 编辑模型列表
vim examples/model_transfer_list.txt

# 执行批量迁移
bash scripts/quick_start_transfer.sh batch examples/model_transfer_list.txt
```

## 📊 脚本特性对比

### `hf_to_modelscope_transfer.py` (完整版)

**✅ 优点:**
- 完整的错误处理和日志记录
- 支持模型验证和完整性检查
- 支持仅验证模式（`--verify-only`）
- 自动检查依赖和认证状态
- 详细的进度显示和统计信息
- 支持自定义提交信息
- 临时文件自动清理

**🎛️ 参数:**
```bash
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "model-name" \
  --ms-model "your-org/model-name" \
  [--hf-token "token"] \
  [--commit-message "message"] \
  [--verify-only] \
  [--log-level DEBUG]
```

### `simple_hf_to_ms.py` (简化版)

**✅ 优点:**
- 代码简洁，易于理解和修改
- 快速执行，适合简单迁移
- 依赖少，启动快
- 适合脚本化和自动化

**🎛️ 参数:**
```bash
python scripts/simple_hf_to_ms.py <hf_model> <ms_model> [hf_token]
```

### `batch_transfer.sh` (批量处理)

**✅ 优点:**
- 支持批量处理多个模型
- 自动统计成功/失败数量
- 内置延迟避免API限制
- 支持注释和空行
- 失败时不中断，继续处理其他模型

**📄 输入格式:**
```
# 注释行
huggingface_model_name modelscope_model_name
microsoft/DialoGPT-small your-org/DialoGPT-small
microsoft/DialoGPT-medium your-org/DialoGPT-medium
```

### `quick_start_transfer.sh` (快速入门)

**✅ 优点:**
- 新手友好，提供向导式操作
- 自动安装依赖
- 环境测试功能
- 统一的命令行界面

**🎛️ 操作:**
- `setup`: 安装依赖和认证指导
- `transfer`: 单模型迁移
- `batch`: 批量迁移
- `test`: 环境测试

## 🔧 依赖要求

### Python包
```bash
pip install transformers huggingface_hub modelscope gitpython
```

### 认证要求
```bash
# Hugging Face
huggingface-cli login --token "hf_your_token"

# ModelScope  
modelscope login --token "your_token"
```

## 📈 使用建议

### 新手用户
1. 使用 `quick_start_transfer.sh setup` 设置环境
2. 使用 `quick_start_transfer.sh test` 测试配置
3. 从小模型开始测试迁移

### 高级用户
1. 直接使用 `simple_hf_to_ms.py` 进行快速迁移
2. 使用 `hf_to_modelscope_transfer.py` 的 `--verify-only` 模式验证大模型
3. 编写自定义批量处理脚本

### 生产环境
1. 使用完整功能脚本进行迁移
2. 启用详细日志记录（`--log-level DEBUG`）
3. 先验证再迁移，确保模型完整性

## ⚠️ 注意事项

### 1. 模型大小
- 小模型（<1GB）：任何脚本都可以快速处理
- 中等模型（1-10GB）：建议先验证再迁移
- 大模型（>10GB）：使用完整功能脚本，监控进度

### 2. 网络环境
- 确保网络稳定，大模型下载可能需要很长时间
- 在网络不稳定时使用 `--verify-only` 模式测试

### 3. 权限管理
- 确保对源模型有读取权限
- 确保对目标ModelScope组织有写入权限
- 私有模型需要提供正确的token

### 4. 存储空间
- 迁移过程中会创建临时文件
- 确保有足够的磁盘空间（至少是模型大小的2倍）

## 🔍 故障排除

### 常见问题解决

1. **依赖缺失**: 运行 `quick_start_transfer.sh setup`
2. **认证失败**: 重新运行登录命令
3. **网络超时**: 检查网络连接，稍后重试
4. **权限不足**: 检查token权限和仓库访问权限
5. **空间不足**: 清理磁盘空间

### 调试工具

```bash
# 环境检查
bash scripts/quick_start_transfer.sh test

# 详细日志
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "model" --ms-model "model" \
  --log-level DEBUG --verify-only
```

## 🎯 总结

这套脚本为不同使用场景提供了完整的解决方案：

- **🚀 快速入门**: `quick_start_transfer.sh`
- **🎯 简单迁移**: `simple_hf_to_ms.py`  
- **🔧 专业控制**: `hf_to_modelscope_transfer.py`
- **📦 批量处理**: `batch_transfer.sh`

选择适合您需求的脚本，享受从Hugging Face到ModelScope的无缝模型迁移体验！