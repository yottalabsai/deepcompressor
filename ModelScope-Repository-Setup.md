# ModelScope仓库创建和上传指南

## 🎯 当前状态
- ✅ 模型已下载：`/root/flux_model_cache/flux-dev` (53.91GB)
- ✅ 用户名确认：`xiaosa` (有效)
- ❌ 需要：在ModelScope上创建仓库并正确上传

## 📋 解决步骤

### 步骤1: 在ModelScope网页上创建仓库

1. **访问创建页面**：
   ```
   https://modelscope.cn/models/create
   ```

2. **填写仓库信息**：
   - **模型名称**：`FLUX.1-dev`
   - **显示名称**：`FLUX.1-dev 文本到图像生成模型`
   - **可见性**：选择合适的选项（公开或私有）
   - **许可证**：选择 `flux-1-dev-non-commercial-license` 或类似
   - **标签**：`text-to-image`, `diffusion`, `flux`

3. **创建仓库**后会得到：
   ```
   https://modelscope.cn/xiaosa/FLUX.1-dev
   ```

### 步骤2: 使用修复脚本上传

在服务器上运行：

```bash
# 方法1: 使用环境变量
export MODELSCOPE_API_TOKEN="dd75741e-ebfb-49fa-b53a-a358a56fd765"
python fix_git_auth_upload.py /root/flux_model_cache/flux-dev xiaosa FLUX.1-dev

# 方法2: 直接提供token
python fix_git_auth_upload.py /root/flux_model_cache/flux-dev xiaosa FLUX.1-dev dd75741e-ebfb-49fa-b53a-a358a56fd765
```

### 步骤3: 手动Git上传（备用方案）

如果脚本仍有问题，可以手动操作：

```bash
cd /root/flux_model_cache/flux-dev

# 配置Git凭据
git config --global credential.helper store
echo 'https://oauth2:dd75741e-ebfb-49fa-b53a-a358a56fd765@www.modelscope.cn' >> ~/.git-credentials

# 初始化仓库
git init
git config user.email "your-email@example.com"
git config user.name "xiaosa"

# 添加文件
git add .
git commit -m "Upload FLUX.1-dev model from Hugging Face"

# 添加远程仓库
git remote add origin https://www.modelscope.cn/xiaosa/FLUX.1-dev.git

# 推送
git push -u origin main
# 如果main失败，尝试master
git push -u origin master
```

## 🔍 常见问题解决

### 问题1: 认证失败
```
HTTP Basic: Access denied
```

**解决方案**：
- 确认Token格式正确
- 使用 `oauth2:token` 格式而不是用户名密码
- 检查Token权限

### 问题2: 分支问题
```
remote: 
remote: The default branch should be named main instead of master
```

**解决方案**：
```bash
git branch -M main
git push -u origin main
```

### 问题3: 大文件上传
```
error: RPC failed; HTTP 413 curl 22 The requested URL returned error: 413
```

**解决方案**：
```bash
git config http.postBuffer 524288000
git config http.maxRequestBuffer 100M
```

## 🎉 成功标志

上传成功后您将看到：
- ✅ 推送成功消息
- 🔗 模型地址：`https://modelscope.cn/xiaosa/FLUX.1-dev`
- 📊 ModelScope页面显示模型文件

## 💡 下一步

上传成功后，可以：
1. 在ModelScope页面查看模型
2. 测试模型下载和使用
3. 分享模型链接
4. 清理临时缓存文件