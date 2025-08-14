#!/bin/bash
# 快速开始：Hugging Face 到 ModelScope 模型迁移

echo "🚀 Hugging Face 到 ModelScope 模型迁移 - 快速开始"
echo "=================================================="

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法:"
    echo "  $0 setup                    # 安装依赖和设置认证"
    echo "  $0 transfer <hf> <ms>       # 迁移单个模型"
    echo "  $0 batch <file>             # 批量迁移"
    echo "  $0 test                     # 测试环境"
    echo ""
    echo "示例:"
    echo "  $0 setup"
    echo "  $0 transfer microsoft/DialoGPT-small your-org/DialoGPT-small"
    echo "  $0 batch examples/model_transfer_list.txt"
    exit 1
fi

ACTION="$1"

case "$ACTION" in
    "setup")
        echo "📦 安装依赖..."
        pip install transformers huggingface_hub modelscope gitpython
        
        echo ""
        echo "🔐 设置认证..."
        echo "请按照提示完成认证："
        
        echo ""
        echo "1️⃣ Hugging Face认证:"
        echo "如果您有token，可以运行："
        echo "   python -m huggingface_hub.commands.huggingface_cli login --token \"hf_your_token\""
        echo "否则运行交互式登录："
        echo "   python -m huggingface_hub.commands.huggingface_cli login"
        
        echo ""
        echo "2️⃣ ModelScope认证:"
        echo "如果您有token，可以运行："
        echo "   python -m modelscope.cli.cli login --token \"your_token\""
        echo "否则运行交互式登录："
        echo "   python -m modelscope.cli.cli login"
        
        echo ""
        echo "✅ 依赖安装完成！请手动完成认证步骤。"
        ;;
        
    "transfer")
        if [ $# -ne 3 ]; then
            echo "❌ 参数错误"
            echo "使用方法: $0 transfer <hf_model> <ms_model>"
            echo "示例: $0 transfer microsoft/DialoGPT-small your-org/DialoGPT-small"
            exit 1
        fi
        
        HF_MODEL="$2"
        MS_MODEL="$3"
        
        echo "🔄 开始迁移模型:"
        echo "   源: $HF_MODEL"
        echo "   目标: $MS_MODEL"
        echo ""
        
        python scripts/simple_hf_to_ms.py "$HF_MODEL" "$MS_MODEL"
        ;;
        
    "batch")
        if [ $# -ne 2 ]; then
            echo "❌ 参数错误"
            echo "使用方法: $0 batch <model_list_file>"
            echo "示例: $0 batch examples/model_transfer_list.txt"
            exit 1
        fi
        
        MODEL_FILE="$2"
        
        if [ ! -f "$MODEL_FILE" ]; then
            echo "❌ 文件不存在: $MODEL_FILE"
            exit 1
        fi
        
        echo "📋 批量迁移模型列表: $MODEL_FILE"
        echo ""
        
        bash scripts/batch_transfer.sh "$MODEL_FILE"
        ;;
        
    "test")
        echo "🧪 测试环境配置..."
        echo ""
        
        echo "1️⃣ 检查Python依赖:"
        for pkg in transformers huggingface_hub modelscope git; do
            if python -c "import $pkg" 2>/dev/null; then
                echo "   ✅ $pkg"
            else
                echo "   ❌ $pkg (未安装)"
            fi
        done
        
        echo ""
        echo "2️⃣ 检查认证状态:"
        python scripts/check_modelscope_auth.py
        
        echo ""
        echo "3️⃣ 测试迁移 (验证模式):"
        echo "   正在测试下载小模型..."
        if python scripts/hf_to_modelscope_transfer.py \
           --hf-model "microsoft/DialoGPT-small" \
           --ms-model "test/DialoGPT-small" \
           --verify-only 2>/dev/null; then
            echo "   ✅ 测试通过，环境配置正确"
        else
            echo "   ❌ 测试失败，请检查网络和认证"
        fi
        ;;
        
    *)
        echo "❌ 未知操作: $ACTION"
        echo "支持的操作: setup, transfer, batch, test"
        exit 1
        ;;
esac