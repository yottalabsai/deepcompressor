#!/bin/bash
# 服务器环境下运行FLUX.1-dev迁移脚本

echo "🚀 服务器环境下的FLUX.1-dev模型迁移"
echo "=================================="

# 检查Python版本
echo "🐍 检查Python版本..."
python --version

# 检查依赖
echo "📦 检查依赖包..."
python -c "
try:
    import transformers, huggingface_hub, modelscope, git
    print('✅ 所有依赖已安装')
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    print('请运行: pip install transformers huggingface_hub modelscope gitpython')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "请先安装依赖包"
    exit 1
fi

# 检查磁盘空间
echo "💾 检查磁盘空间..."
AVAILABLE_SPACE=$(df /tmp | tail -1 | awk '{print $4}')
REQUIRED_SPACE=52428800  # 50GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "⚠️  警告: 可用磁盘空间不足50GB，建议清理后再运行"
    echo "当前可用空间: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
else
    echo "✅ 磁盘空间充足: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
fi

# 检查网络连接
echo "🌐 检查网络连接..."
if ping -c 1 huggingface.co > /dev/null 2>&1; then
    echo "✅ Hugging Face 连接正常"
else
    echo "❌ 无法连接到 Hugging Face"
fi

if ping -c 1 modelscope.cn > /dev/null 2>&1; then
    echo "✅ ModelScope 连接正常"
else
    echo "❌ 无法连接到 ModelScope"
fi

# 检查认证
echo "🔐 检查认证状态..."

# 检查HF认证
python -c "
try:
    from huggingface_hub import whoami
    user = whoami()
    print(f'✅ Hugging Face已认证: {user.get(\"name\", \"Unknown\")}')
except Exception as e:
    print('❌ Hugging Face未认证，请运行:')
    print('   export HF_TOKEN=\"your_token\"')
    print('   或 python -m huggingface_hub.commands.huggingface_cli login --token \"your_token\"')
"

# 检查ModelScope认证
python -c "
import os
try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    if os.environ.get('MODELSCOPE_API_TOKEN'):
        print('✅ ModelScope token已设置')
    else:
        print('❌ ModelScope未认证，请运行:')
        print('   export MODELSCOPE_API_TOKEN=\"your_token\"')
        print('   或 python -m modelscope.cli.cli login --token \"your_token\"')
except Exception as e:
    print(f'❌ ModelScope检查失败: {e}')
"

echo ""
echo "如果所有检查都通过，可以运行迁移脚本："
echo "  python transfer_flux_dev.py"
echo ""
echo "建议在screen或tmux中运行："
echo "  screen -S flux_transfer"
echo "  python transfer_flux_dev.py"
echo ""
echo "监控进度："
echo "  watch -n 10 'df -h /tmp && ps aux | grep python'"