#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLUX.1-dev 模型迁移脚本
从 Hugging Face 迁移到 ModelScope
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# ModelScope token需要通过环境变量或命令行登录设置
# 使用方法：export MODELSCOPE_API_TOKEN="your_token" 或 python -m modelscope.cli.cli login --token "your_token"

def download_flux_from_hf():
    """从Hugging Face下载FLUX.1-dev模型"""
    try:
        from huggingface_hub import snapshot_download
        
        print("📥 开始从Hugging Face下载FLUX.1-dev模型...")
        print("⚠️  注意：这是一个大模型（~23GB），下载可能需要较长时间")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="flux_transfer_")
        local_path = os.path.join(temp_dir, "flux-dev")
        
        print(f"📁 下载目录: {local_path}")
        
        # 下载模型，使用已登录的token
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
        
        print("✅ 下载完成！")
        return temp_dir, local_path
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None, None

def verify_model(local_path):
    """验证模型文件"""
    print("🔍 验证模型文件...")
    
    path_obj = Path(local_path)
    
    # 检查重要文件
    important_files = [
        "config.json",
        "model_index.json"
    ]
    
    model_files = list(path_obj.rglob("*.safetensors")) + list(path_obj.rglob("*.bin"))
    
    print(f"📊 模型验证结果:")
    
    for file in important_files:
        file_path = path_obj / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (缺失)")
    
    print(f"  📦 模型文件数量: {len(model_files)}")
    
    total_size = sum(f.stat().st_size for f in path_obj.rglob("*") if f.is_file())
    print(f"  📏 总大小: {total_size / (1024**3):.2f} GB")
    
    return len(model_files) > 0

def upload_to_modelscope(local_path, ms_model_name="your-org/FLUX.1-dev"):
    """上传到ModelScope"""
    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.repository import Repository
        import git
        
        print(f"📤 开始上传到ModelScope: {ms_model_name}")
        
        api = HubApi()
        
        # 创建仓库（如果不存在）
        try:
            print("🔨 创建ModelScope仓库...")
            api.create_model(
                model_id=ms_model_name,
                visibility=1,  # 公开
                license='flux-1-dev-non-commercial-license',
                chinese_name="FLUX.1-dev 文本到图像生成模型"
            )
            print(f"✅ 仓库创建成功: {ms_model_name}")
        except Exception as e:
            print(f"ℹ️  仓库可能已存在: {e}")
        
        # 克隆仓库
        print("📥 克隆ModelScope仓库...")
        repo = Repository(
            local_dir=local_path,
            model_id=ms_model_name,
            clone_from=f"https://www.modelscope.cn/{ms_model_name}.git"
        )
        
        # 检查git状态并提交
        repo_git = git.Repo(local_path)
        
        print("📋 检查文件变更...")
        if repo_git.is_dirty() or repo_git.untracked_files:
            print("📝 添加文件到git...")
            repo_git.git.add('.')
            
            print("💾 提交变更...")
            commit_message = "Upload FLUX.1-dev model from Hugging Face\n\nFLUX.1-dev is a 12B parameter text-to-image diffusion model."
            repo_git.index.commit(commit_message)
            
            print("🚀 推送到ModelScope...")
            origin = repo_git.remote('origin')
            origin.push()
            
            print(f"🎉 上传成功！")
            print(f"🔗 ModelScope链接: https://modelscope.cn/models/{ms_model_name}")
            return True
        else:
            print("ℹ️  没有变更需要上传")
            return True
            
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 FLUX.1-dev 模型迁移开始")
    print("=" * 50)
    
    # 使用默认的ModelScope目标仓库名
    ms_model_name = "xiaosa/FLUX.1-dev"
    print(f"目标ModelScope仓库: {ms_model_name}")
    
    temp_dir = None
    try:
        # 1. 下载模型
        temp_dir, local_path = download_flux_from_hf()
        if not local_path:
            return False
        
        # 2. 验证模型
        if not verify_model(local_path):
            print("❌ 模型验证失败")
            return False
        
        # 3. 上传到ModelScope
        if not upload_to_modelscope(local_path, ms_model_name):
            return False
        
        print("🎉 FLUX.1-dev 模型迁移完成！")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
        return False
    except Exception as e:
        print(f"💥 意外错误: {e}")
        return False
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            print(f"🧹 清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)