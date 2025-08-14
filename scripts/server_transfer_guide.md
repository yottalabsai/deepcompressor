# 服务器环境下的模型迁移指南

## 🚀 在服务器上运行模型迁移

### 1. 准备工作

#### 安装依赖
```bash
pip install transformers huggingface_hub modelscope gitpython
```

#### 设置认证
```bash
# 方式1: 使用环境变量
export HF_TOKEN="hf_your_token_here"
export MODELSCOPE_API_TOKEN="your_modelscope_token"

# 方式2: 使用命令行登录
python -m huggingface_hub.commands.huggingface_cli login --token "hf_your_token"
python -m modelscope.cli.cli login --token "your_modelscope_token"
```

### 2. 下载脚本到服务器
```bash
# 将迁移脚本上传到服务器
scp transfer_flux_dev.py your_server:/path/to/
scp scripts/hf_to_modelscope_transfer.py your_server:/path/to/
scp scripts/simple_hf_to_ms.py your_server:/path/to/
```

### 3. 运行迁移

#### 选项1: 使用专用FLUX脚本
```bash
python transfer_flux_dev.py
```

#### 选项2: 使用通用脚本
```bash
# 完整功能版本
python scripts/hf_to_modelscope_transfer.py \
  --hf-model "black-forest-labs/FLUX.1-dev" \
  --ms-model "xiaosa/FLUX.1-dev" \
  --log-level INFO

# 简化版本
python scripts/simple_hf_to_ms.py \
  black-forest-labs/FLUX.1-dev \
  xiaosa/FLUX.1-dev
```

### 4. 监控进度

#### 使用screen或tmux运行长时间任务
```bash
# 使用screen
screen -S flux_transfer
python transfer_flux_dev.py
# Ctrl+A, D 分离会话
# screen -r flux_transfer 重新连接

# 使用tmux
tmux new-session -d -s flux_transfer 'python transfer_flux_dev.py'
tmux attach-session -t flux_transfer
```

#### 监控系统资源
```bash
# 监控磁盘空间
df -h

# 监控网络使用
iftop

# 监控进程
htop
```

### 5. 预期时间和资源

| 模型 | 大小 | 预期下载时间 | 预期上传时间 | 磁盘空间需求 |
|------|------|-------------|-------------|--------------|
| FLUX.1-dev | ~23.8GB | 30-120分钟 | 30-120分钟 | ~50GB |

**注意事项：**
- 时间取决于网络带宽
- 建议至少50GB可用磁盘空间
- 临时文件会在迁移完成后自动清理

### 6. 故障排除

#### 常见问题
```bash
# 检查认证状态
python -c "
from huggingface_hub import whoami
print('HF用户:', whoami())
"

python -c "
from modelscope.hub.api import HubApi
api = HubApi()
print('ModelScope认证设置完成')
"

# 检查磁盘空间
df -h /tmp

# 检查网络连接
ping huggingface.co
ping modelscope.cn
```

#### 断点续传
如果下载中断，脚本会自动清理临时文件。重新运行即可从头开始。

### 7. 批量迁移多个模型

创建模型列表文件：
```bash
cat > model_list.txt << EOF
black-forest-labs/FLUX.1-dev xiaosa/FLUX.1-dev
microsoft/DialoGPT-medium xiaosa/DialoGPT-medium
sentence-transformers/all-MiniLM-L6-v2 xiaosa/all-MiniLM-L6-v2
EOF

# 批量处理
while read hf_model ms_model; do
    echo "开始迁移: $hf_model -> $ms_model"
    python scripts/simple_hf_to_ms.py "$hf_model" "$ms_model"
    echo "完成迁移: $ms_model"
    echo "等待30秒..."
    sleep 30
done < model_list.txt
```

### 8. 安全建议

- 不要在脚本中硬编码token
- 使用环境变量或安全的配置文件
- 定期更换access token
- 在公共服务器上注意文件权限

```bash
# 设置环境变量文件权限
chmod 600 ~/.bashrc
chmod 600 ~/.profile

# 临时设置环境变量（会话结束后失效）
export HF_TOKEN="your_token"
export MODELSCOPE_API_TOKEN="your_token"
```