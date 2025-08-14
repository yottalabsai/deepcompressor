# ModelScope支持文档

DeepCompressor现在支持从ModelScope下载和加载模型进行压缩处理。

## 安装和认证

### 1. 安装ModelScope

```bash
pip install modelscope
```

或者安装带有ModelScope支持的完整依赖：

```bash
pip install deepcompressor-toolkit[modelscope]
```

### 2. 认证（可选但推荐）

对于私有模型或更好的下载体验，建议先登录：

```bash
# 交互式登录
modelscope login

# 或直接传入token（适合自动化脚本）
modelscope login --token "your_token_here"
```

或设置环境变量：

```bash
export MODELSCOPE_API_TOKEN="your_token_here"
```

详细认证指南请参考：[ModelScope认证指南](ModelScope-Authentication.md)

## 配置方式

### 1. 通过配置文件启用ModelScope

#### LLM模型配置示例

```yaml
model:
  name: qwen-7b-chat
  use_modelscope: true
  path: "qwen/Qwen-7B-Chat"  # ModelScope模型ID
  modelscope_cache_dir: "~/.cache/modelscope"  # 可选：指定缓存目录
  dtype: torch.float16
  fast_tokenizer: true

quant:
  # 量化配置...
```

#### Diffusion模型配置示例

```yaml
pipeline:
  name: flux.1-dev
  use_modelscope: true
  path: "modelscope/flux.1-dev"  # ModelScope模型ID
  modelscope_cache_dir: "~/.cache/modelscope"  # 可选：指定缓存目录
  dtype: torch.bfloat16

quant:
  # 量化配置...
```

### 2. 自动检测ModelScope模型

如果模型路径看起来像ModelScope模型ID（包含组织名/模型名格式），系统会自动启用ModelScope支持：

```yaml
model:
  name: qwen-7b-chat
  path: "qwen/Qwen-7B-Chat"  # 自动检测为ModelScope模型ID
```

## 支持的配置参数

- `use_modelscope`: 布尔值，是否强制使用ModelScope加载模型
- `modelscope_cache_dir`: 字符串，ModelScope模型缓存目录，默认使用ModelScope的默认缓存位置
- `path`: 模型路径，当使用ModelScope时应为ModelScope模型ID

注意：认证信息通过ModelScope CLI或环境变量管理，不需要在配置文件中指定。

## 使用方式

### 1. LLM模型压缩

```bash
python -m deepcompressor.app.llm.ptq --config examples/llm/configs/qwen-modelscope.yaml
```

### 2. Diffusion模型压缩

```bash
python -m deepcompressor.app.diffusion.ptq --config examples/diffusion/configs/model/flux.1-dev-modelscope.yaml
```

## 注意事项

1. **网络连接**：首次使用需要网络连接来下载模型，后续会使用本地缓存
2. **存储空间**：ModelScope模型会被下载到本地缓存目录，确保有足够的存储空间
3. **模型ID格式**：ModelScope模型ID通常为 `organization/model-name` 格式
4. **兼容性**：ModelScope支持与现有的HuggingFace加载方式完全兼容，可以在同一项目中混合使用

## 故障排除

### 1. ImportError: ModelScope library is required

解决方案：安装modelscope库
```bash
pip install modelscope
```

### 2. 认证相关错误

```
Error: Invalid token / Access denied
```

解决方案：
```bash
# 重新登录
modelscope login

# 或检查环境变量
echo $MODELSCOPE_API_TOKEN
```

### 3. 模型下载失败

可能原因：
- 网络连接问题
- 模型ID不正确
- 权限问题（私有模型需要认证）

解决方案：
- 检查网络连接
- 验证ModelScope模型ID的正确性
- 对于私有模型，确保已登录且有权限访问

### 3. 缓存目录问题

如果遇到缓存目录相关问题，可以手动指定缓存目录：

```yaml
modelscope_cache_dir: "/path/to/your/cache"
```

## 示例模型

以下是一些可以用于测试的ModelScope模型示例：

### LLM模型
- `qwen/Qwen-7B-Chat`
- `baichuan-inc/Baichuan2-7B-Chat`
- `ZhipuAI/chatglm3-6b`

### Diffusion模型
- 根据ModelScope实际可用的diffusion模型进行配置

注意：实际使用时请替换为真实存在的ModelScope模型ID。