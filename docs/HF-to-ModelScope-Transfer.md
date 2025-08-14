# Hugging Face 到 ModelScope 模型迁移指南

本指南介绍如何使用提供的脚本将模型从Hugging Face迁移到ModelScope。

## 📋 前置准备

### 1. 安装依赖

```bash
pip install transformers huggingface_hub modelscope gitpython
```

### 2. 认证设置

#### Hugging Face认证
```bash
# 交互式登录
huggingface-cli login

# 或使用token
huggingface-cli login --token "hf_your_token_here"
```

#### ModelScope认证
```bash
# 交互式登录
modelscope login

# 或使用token
modelscope login --token "your_modelscope_token"
```

## 🚀 使用方法

### 方式1: 完整功能脚本

适合复杂场景，功能完整，支持验证和错误处理。

```bash
# 基本用法
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "microsoft/DialoGPT-medium" \
  --ms-model "your-org/DialoGPT-medium"

# 使用私有模型token
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "private-org/private-model" \
  --ms-model "your-org/private-model" \
  --hf-token "hf_your_private_token"

# 仅验证不上传
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "microsoft/DialoGPT-medium" \
  --ms-model "your-org/DialoGPT-medium" \
  --verify-only

# 自定义提交信息
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "microsoft/DialoGPT-medium" \
  --ms-model "your-org/DialoGPT-medium" \
  --commit-message "Initial model upload from HF"
```

#### 完整参数说明

| 参数 | 说明 | 必需 | 示例 |
|-----|------|------|------|
| `--hf-model` | Hugging Face模型名称 | ✅ | `microsoft/DialoGPT-medium` |
| `--ms-model` | ModelScope模型名称 | ✅ | `your-org/DialoGPT-medium` |
| `--hf-token` | HF访问token（私有模型） | ❌ | `hf_xxxxxxxxxxxx` |
| `--commit-message` | 提交信息 | ❌ | `"Upload from HF"` |
| `--verify-only` | 仅验证不上传 | ❌ | - |
| `--log-level` | 日志级别 | ❌ | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### 方式2: 简化脚本

适合快速迁移，代码简洁。

```bash
# 基本用法
python scripts/simple_hf_to_ms.py microsoft/DialoGPT-medium your-org/DialoGPT-medium

# 使用私有token
python scripts/simple_hf_to_ms.py private-org/model your-org/model hf_your_token
```

### 方式3: 批量迁移

适合迁移多个模型。

```bash
# 1. 创建模型列表文件
cat > my_models.txt << EOF
microsoft/DialoGPT-small your-org/DialoGPT-small
microsoft/DialoGPT-medium your-org/DialoGPT-medium
sentence-transformers/all-MiniLM-L6-v2 your-org/all-MiniLM-L6-v2
EOF

# 2. 执行批量迁移
bash scripts/batch_transfer.sh my_models.txt
```

## 📊 支持的模型类型

| 模型类型 | 示例 | 特殊说明 |
|---------|------|----------|
| **语言模型** | GPT、BERT、T5 | 支持各种大小的LLM |
| **对话模型** | DialoGPT、BlenderBot | 包含对话历史处理 |
| **代码模型** | CodeBERT、CodeT5 | 程序理解和生成 |
| **嵌入模型** | Sentence-Transformers | 文本相似度计算 |
| **视觉模型** | ViT、CLIP | 图像分类和理解 |
| **多模态模型** | CLIP、BLIP | 图文理解 |
| **Diffusion模型** | Stable Diffusion | 图像生成 |

## 🔧 高级用法

### 1. 条件迁移

```bash
# 创建条件迁移脚本
cat > conditional_transfer.sh << 'EOF'
#!/bin/bash
MODEL_LIST="models.txt"
MIN_SIZE_MB=100  # 只迁移大于100MB的模型

while read hf_model ms_model; do
    # 先验证模型大小
    if python scripts/hf_to_modelscope_transfer.py \
       --hf-model "$hf_model" \
       --ms-model "$ms_model" \
       --verify-only | grep -q "总大小.*[0-9][0-9][0-9]\+"; then
        echo "✅ 模型足够大，开始迁移: $hf_model"
        python scripts/simple_hf_to_ms.py "$hf_model" "$ms_model"
    else
        echo "⏭️  跳过小模型: $hf_model"
    fi
done < "$MODEL_LIST"
EOF

chmod +x conditional_transfer.sh
```

### 2. 增量同步

```bash
# 检查ModelScope仓库是否存在，只迁移新模型
cat > incremental_sync.py << 'EOF'
import requests
import sys

def check_ms_model_exists(model_name):
    url = f"https://modelscope.cn/api/v1/models/{model_name}"
    response = requests.get(url)
    return response.status_code == 200

def main():
    with open('models.txt', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                hf_model, ms_model = line.strip().split()
                if not check_ms_model_exists(ms_model):
                    print(f"🆕 新模型，开始迁移: {hf_model}")
                    # 执行迁移...
                else:
                    print(f"✅ 模型已存在，跳过: {ms_model}")

if __name__ == "__main__":
    main()
EOF
```

### 3. 并行迁移

```bash
# 使用GNU parallel并行处理（谨慎使用，避免API限制）
cat models.txt | parallel -j 2 --colsep ' ' \
  python scripts/simple_hf_to_ms.py {1} {2}
```

## ⚠️ 注意事项

### 1. 权限和认证
- 确保对源模型有读取权限
- 确保对目标组织有写入权限
- 私有模型需要相应的访问token

### 2. 模型大小限制
- 大模型（>10GB）可能需要较长下载时间
- 注意本地磁盘空间，临时文件可能很大
- ModelScope对单个文件大小可能有限制

### 3. 网络和速度
- 建议在网络良好的环境下进行迁移
- 可以使用 `--verify-only` 先测试下载
- 批量迁移时注意API频率限制

### 4. 版本控制
- 迁移会创建新的git提交
- 可以自定义提交信息
- 建议在迁移前备份重要数据

## 🔍 故障排除

### 常见错误

#### 1. 认证失败
```
Error: Invalid token
```
**解决**: 重新运行登录命令
```bash
huggingface-cli login
modelscope login --token "your_token"
```

#### 2. 模型不存在
```
Error: Repository not found
```
**解决**: 检查模型名称拼写，确认模型存在

#### 3. 权限不足
```
Error: You don't have permission to access this repository
```
**解决**: 检查私有模型的访问权限，提供正确的token

#### 4. 网络超时
```
Error: Connection timeout
```
**解决**: 检查网络连接，稍后重试

#### 5. 磁盘空间不足
```
Error: No space left on device
```
**解决**: 清理磁盘空间，特别是临时目录

### 调试技巧

```bash
# 启用详细日志
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "microsoft/DialoGPT-medium" \
  --ms-model "your-org/DialoGPT-medium" \
  --log-level DEBUG

# 仅验证模型完整性
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "microsoft/DialoGPT-medium" \
  --ms-model "your-org/DialoGPT-medium" \
  --verify-only

# 检查认证状态
python scripts/check_modelscope_auth.py
```

## 📈 最佳实践

1. **先测试小模型**: 从小模型开始，确保流程正确
2. **使用验证模式**: 先用 `--verify-only` 检查模型完整性
3. **批量处理**: 对于多个模型，使用批量脚本提高效率
4. **监控过程**: 大模型迁移时监控网络和磁盘使用情况
5. **备份重要数据**: 迁移前备份重要的模型和配置
6. **合理命名**: 在ModelScope上使用清晰的模型命名规范

## 🔗 相关链接

- [Hugging Face Hub文档](https://huggingface.co/docs/hub/index)
- [ModelScope文档](https://modelscope.cn/docs/)
- [ModelScope认证指南](ModelScope-Authentication.md)
- [DeepCompressor ModelScope支持](ModelScope-Support.md)