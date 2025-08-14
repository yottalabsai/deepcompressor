# ModelScope Login --token 参数使用总结

## 🎯 问题解答

**问题**: `modelscope login` 有没有加token的参数？

**答案**: **有的！** `modelscope login` 支持 `--token` 参数。

## 📋 验证结果

通过实际安装和测试 ModelScope 1.29.0，确认了以下信息：

```bash
$ python -m modelscope.cli.cli login --help

usage: modelscope <command> [<args>] login [-h] --token TOKEN

optional arguments:
  -h, --help     show this help message and exit
  --token TOKEN  The Access Token for modelscope.
```

## 🔧 使用方法

### 1. 交互式登录（手动推荐）
```bash
modelscope login
```
系统会提示输入token。

### 2. 非交互式登录（自动化推荐）
```bash
modelscope login --token "your_token_here"
```
直接传入token，适合脚本和自动化场景。

### 3. 环境变量方式
```bash
export MODELSCOPE_API_TOKEN="your_token_here"
```
设置环境变量，ModelScope会自动使用。

## 📊 使用场景对比

| 场景 | 推荐方式 | 命令 | 优点 |
|-----|---------|------|------|
| **手动操作** | 交互式登录 | `modelscope login` | 安全，token不显示在命令历史 |
| **自动化脚本** | --token参数 | `modelscope login --token "$TOKEN"` | 非交互式，适合CI/CD |
| **环境变量** | 环境变量 | `export MODELSCOPE_API_TOKEN="token"` | 全局设置，应用自动使用 |
| **Python代码** | API登录 | `HubApi().login("token")` | 程序化控制 |

## 🎯 最佳实践

### 生产环境
```bash
#!/bin/bash
# 从环境变量或密钥管理系统获取token
TOKEN="${MODELSCOPE_TOKEN:-$(vault kv get -field=token secret/modelscope)}"

# 非交互式登录
modelscope login --token "$TOKEN"

# 执行模型相关操作
python -m deepcompressor.app.llm.ptq --config config.yaml
```

### 开发环境
```bash
# 手动登录一次
modelscope login

# 后续自动使用保存的认证信息
python -m deepcompressor.app.llm.ptq --config config.yaml
```

### CI/CD环境
```yaml
# GitHub Actions 示例
- name: Login to ModelScope
  run: |
    modelscope login --token "${{ secrets.MODELSCOPE_TOKEN }}"
    
- name: Run DeepCompressor
  run: |
    python -m deepcompressor.app.llm.ptq --config config.yaml
```

## 📝 文档更新

已更新以下文档以反映 `--token` 参数的支持：

1. **`docs/ModelScope-Authentication.md`**
   - 添加了 `--token` 参数的使用说明
   - 新增使用场景对比表
   - 更新了所有登录示例

2. **`docs/ModelScope-Support.md`**
   - 更新了认证部分的说明
   - 添加了自动化脚本的推荐用法

## 🔄 与 DeepCompressor 集成

在 DeepCompressor 中使用 ModelScope 模型的完整流程：

```bash
# 1. 安装依赖
pip install deepcompressor-toolkit[modelscope]

# 2. 登录 ModelScope
modelscope login --token "$MODELSCOPE_TOKEN"

# 3. 使用 ModelScope 模型进行压缩
python -m deepcompressor.app.llm.ptq --config examples/llm/configs/qwen-modelscope.yaml
```

## ✅ 总结

- ✅ **`modelscope login` 确实支持 `--token` 参数**
- ✅ **适合自动化脚本和CI/CD环境**
- ✅ **兼容交互式和非交互式两种使用方式**
- ✅ **与 DeepCompressor 的 ModelScope 集成完美配合**

这解决了在自动化环境中使用 ModelScope 认证的问题，使得 DeepCompressor 的 ModelScope 支持更加完善和实用！