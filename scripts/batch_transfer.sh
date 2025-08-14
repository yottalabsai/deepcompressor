#!/bin/bash
# 批量迁移脚本
# 使用方法: bash scripts/batch_transfer.sh models.txt

set -e

if [ $# -eq 0 ]; then
    echo "使用方法: $0 <模型列表文件>"
    echo ""
    echo "模型列表文件格式（每行一个模型对）:"
    echo "huggingface_model_name modelscope_model_name"
    echo ""
    echo "示例:"
    echo "microsoft/DialoGPT-small your-org/DialoGPT-small"
    echo "microsoft/DialoGPT-medium your-org/DialoGPT-medium"
    exit 1
fi

MODEL_LIST_FILE="$1"

if [ ! -f "$MODEL_LIST_FILE" ]; then
    echo "❌ 文件不存在: $MODEL_LIST_FILE"
    exit 1
fi

echo "🚀 开始批量迁移..."
echo "📄 模型列表文件: $MODEL_LIST_FILE"
echo ""

SUCCESS_COUNT=0
FAILURE_COUNT=0
TOTAL_COUNT=0

# 读取模型列表并处理
while IFS=' ' read -r hf_model ms_model || [ -n "$hf_model" ]; do
    # 跳过空行和注释行
    if [[ -z "$hf_model" || "$hf_model" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    echo "📦 处理模型 $TOTAL_COUNT: $hf_model -> $ms_model"
    echo "----------------------------------------"
    
    # 执行迁移
    if python3 scripts/simple_hf_to_ms.py "$hf_model" "$ms_model"; then
        echo "✅ 成功: $hf_model -> $ms_model"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ 失败: $hf_model -> $ms_model"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi
    
    echo ""
    
    # 添加延迟避免请求过于频繁
    if [ $TOTAL_COUNT -lt $(wc -l < "$MODEL_LIST_FILE") ]; then
        echo "⏳ 等待 5 秒..."
        sleep 5
    fi
    
done < "$MODEL_LIST_FILE"

# 输出统计结果
echo "========================================"
echo "📊 批量迁移完成!"
echo "   总计: $TOTAL_COUNT"
echo "   成功: $SUCCESS_COUNT"
echo "   失败: $FAILURE_COUNT"
echo "========================================"

if [ $FAILURE_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi