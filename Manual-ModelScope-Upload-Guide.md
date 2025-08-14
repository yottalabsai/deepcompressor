# ModelScope手动上传指南

## 🚨 问题分析

从您的错误信息可以看出两个主要问题：

1. **Namespace无效**: `xiaosa` 这个用户名在ModelScope上可能不存在或不合法
2. **API兼容性**: `Repository.__init__()` 不支持 `local_dir` 参数

## 🔧 解决方案

### 方案1: 检查并修正用户名

#### 1. 确认您的ModelScope用户名
访问：https://modelscope.cn/profile

查看您的实际用户名，可能不是 `xiaosa`。

#### 2. 常见用户名问题
- ModelScope用户名可能包含数字或特殊字符
- 需要使用确切的用户名或组织名
- 某些用户名可能被保留

### 方案2: 使用修复版脚本

```bash
# 在服务器上运行修复版上传脚本
python fix_modelscope_upload.py /root/flux_model_cache/flux-dev your-real-username/FLUX.1-dev
```

### 方案3: 手动创建仓库后上传

#### 步骤1: 在ModelScope网页上创建仓库
1. 访问：https://modelscope.cn/models/create
2. 手动创建名为 `FLUX.1-dev` 的仓库
3. 选择合适的许可证（如Apache 2.0）
4. 获取准确的仓库URL

#### 步骤2: 使用Git手动上传
```bash
# 进入模型目录
cd /root/flux_model_cache/flux-dev

# 初始化git（如果还没有）
git init

# 设置远程仓库（替换为您的实际用户名）
git remote add origin https://www.modelscope.cn/your-username/FLUX.1-dev.git

# 添加所有文件
git add .

# 提交
git commit -m "Upload FLUX.1-dev model from Hugging Face"

# 推送（可能需要输入认证信息）
git push -u origin main
```

### 方案4: 更新ModelScope库

```bash
# 更新到最新版本
pip install --upgrade modelscope

# 检查版本
python -c "import modelscope; print(modelscope.__version__)"
```

## 🎯 推荐的完整解决流程

### 1. 首先确认用户名
```bash
# 检查当前ModelScope配置
python -c "
from modelscope.hub.api import HubApi
api = HubApi()
print('请访问 https://modelscope.cn/profile 确认您的用户名')
"
```

### 2. 手动创建仓库
- 访问：https://modelscope.cn/models/create
- 创建仓库：`FLUX.1-dev`
- 记录完整的仓库路径

### 3. 使用修复版脚本上传
```bash
python fix_modelscope_upload.py /root/flux_model_cache/flux-dev correct-username/FLUX.1-dev
```

## 📋 常见错误解决

### 错误1: "namespace is not valid"
**原因**: 用户名不正确或不存在
**解决**: 
1. 访问 https://modelscope.cn/profile 确认用户名
2. 检查是否有权限创建仓库
3. 尝试使用完整的组织名

### 错误2: "Repository.__init__() got an unexpected keyword argument 'local_dir'"
**原因**: ModelScope库版本不兼容
**解决**:
1. 更新ModelScope: `pip install --upgrade modelscope`
2. 使用修复版脚本（避免使用Repository类）
3. 直接使用git命令

### 错误3: 权限问题
**原因**: Token无效或权限不足
**解决**:
1. 重新生成ModelScope token
2. 确保token有创建仓库的权限
3. 检查环境变量是否正确设置

## 💡 最简单的解决方法

如果以上方法都有问题，推荐使用最简单的方法：

1. **在ModelScope网页上手动创建仓库**
2. **使用修复版脚本上传文件**
3. **如果还是失败，直接使用git命令**

```bash
# 最简单的上传命令
cd /root/flux_model_cache/flux-dev
git init
git add .
git commit -m "Upload FLUX.1-dev model"
git remote add origin https://www.modelscope.cn/your-username/FLUX.1-dev.git
git push -u origin main
```

## 🔗 相关链接

- ModelScope用户中心: https://modelscope.cn/profile
- 创建模型仓库: https://modelscope.cn/models/create
- ModelScope文档: https://modelscope.cn/docs/
- Token管理: https://modelscope.cn/my/myaccesstoken

模型已经成功下载到 `/root/flux_model_cache/flux-dev`，现在只需要解决上传的认证和API问题就可以完成迁移了！