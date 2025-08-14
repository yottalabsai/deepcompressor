#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 Hugging Face 到 ModelScope 模型迁移脚本

适合快速迁移小型模型，功能精简但易用。

使用方法:
python scripts/simple_hf_to_ms.py microsoft/DialoGPT-medium your-org/DialoGPT-medium
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

def check_model_exists(local_path):
    """检查模型是否已存在"""
    if not os.path.exists(local_path):
        return False
    
    # 检查目录是否为空
    if not os.listdir(local_path):
        return False
    
    # 检查是否有基本文件
    basic_files = ["README.md", "config.json", "model_index.json"]
    has_basic = any(os.path.exists(os.path.join(local_path, f)) for f in basic_files)
    
    if not has_basic:
        return False
    
    print(f"✅ 模型已存在: {local_path}")
    return True

def download_from_hf(hf_model, local_path, token=None):
    """从Hugging Face下载模型"""
    try:
        from huggingface_hub import snapshot_download
        
        # 检查模型是否已存在
        if check_model_exists(local_path):
            print(f"⏭️  跳过下载，模型已存在: {local_path}")
            return True
        
        print(f"📥 正在从Hugging Face下载: {hf_model}")
        
        kwargs = {
            "repo_id": hf_model,
            "local_dir": local_path,
            "local_dir_use_symlinks": False
        }
        if token:
            kwargs["token"] = token
            
        snapshot_download(**kwargs)
        print(f"✅ 下载完成: {local_path}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def upload_to_modelscope(local_path, ms_model, commit_msg="Upload from Hugging Face"):
    """上传到ModelScope"""
    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.repository import Repository
        import git
        
        print(f"📤 正在上传到ModelScope: {ms_model}")
        
        # 创建仓库（如果不存在）
        api = HubApi()
        try:
            api.create_model(model_id=ms_model, visibility=1)
            print(f"✅ 创建仓库: {ms_model}")
        except:
            print(f"ℹ️  仓库可能已存在: {ms_model}")
        
        # 克隆并推送
        repo = Repository(
            local_dir=local_path,
            model_id=ms_model,
            clone_from=f"https://www.modelscope.cn/{ms_model}.git"
        )
        
        repo_git = git.Repo(local_path)
        if repo_git.is_dirty() or repo_git.untracked_files:
            repo_git.git.add('.')
            repo_git.index.commit(commit_msg)
            origin = repo_git.remote('origin')
            origin.push()
            print(f"✅ 上传完成: https://modelscope.cn/models/{ms_model}")
        else:
            print("ℹ️  没有变更需要上传")
        
        return True
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("使用方法: python simple_hf_to_ms.py <hf_model> <ms_model> [hf_token]")
        print("示例: python simple_hf_to_ms.py microsoft/DialoGPT-medium your-org/DialoGPT-medium")
        sys.exit(1)
    
    hf_model = sys.argv[1]
    ms_model = sys.argv[2]
    hf_token = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"🚀 开始迁移: {hf_model} -> {ms_model}")
    
    # 使用持久缓存目录
    cache_root = os.path.expanduser("~/.cache/simple_model_transfer")
    os.makedirs(cache_root, exist_ok=True)
    safe_model_name = hf_model.replace("/", "_").replace("-", "_")
    local_path = os.path.join(cache_root, safe_model_name)
    
    try:
        # 下载
        if not download_from_hf(hf_model, local_path, hf_token):
            sys.exit(1)
        
        # 验证
        model_files = list(Path(local_path).rglob("*.bin")) + list(Path(local_path).rglob("*.safetensors"))
        print(f"📊 发现 {len(model_files)} 个模型文件")
        
        # 上传
        if not upload_to_modelscope(local_path, ms_model):
            sys.exit(1)
        
        print("🎉 迁移完成！")
        print(f"📁 模型已缓存到: {local_path}")
        print("💡 下次运行相同模型将跳过下载")
    
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"💥 意外错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()