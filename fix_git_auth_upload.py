#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复ModelScope Git认证和上传问题
"""

import os
import sys
import subprocess
import tempfile
import shutil

def setup_git_credentials(token):
    """设置Git凭据"""
    print("🔧 配置Git凭据...")
    
    try:
        # 设置Git凭据存储
        subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=True)
        
        # 创建凭据文件
        home = os.path.expanduser('~')
        git_credentials_path = os.path.join(home, '.git-credentials')
        
        # 添加ModelScope凭据
        with open(git_credentials_path, 'a') as f:
            f.write(f'https://oauth2:{token}@www.modelscope.cn\n')
        
        print("✅ Git凭据配置成功")
        return True
        
    except Exception as e:
        print(f"❌ Git凭据配置失败: {e}")
        return False

def create_and_upload_to_modelscope(local_path, username, model_name, token):
    """创建仓库并上传模型"""
    print(f"📤 上传模型到ModelScope: {username}/{model_name}")
    
    # 临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = os.path.join(temp_dir, 'repo')
        
        try:
            # 1. 设置Git凭据
            if not setup_git_credentials(token):
                return False
            
            # 2. 创建本地Git仓库
            print(f"📁 准备本地仓库...")
            shutil.copytree(local_path, repo_path)
            
            os.chdir(repo_path)
            
            # 初始化Git仓库
            subprocess.run(['git', 'init'], check=True)
            subprocess.run(['git', 'config', 'user.email', 'automated@modelscope.cn'], check=True)
            subprocess.run(['git', 'config', 'user.name', username], check=True)
            
            # 3. 添加文件
            print("📦 添加模型文件...")
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', f'Upload {model_name} model from Hugging Face'], check=True)
            
            # 4. 设置远程仓库
            remote_url = f'https://www.modelscope.cn/{username}/{model_name}.git'
            print(f"🔗 设置远程仓库: {remote_url}")
            subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True)
            
            # 5. 推送到远程
            print("⬆️ 推送到ModelScope...")
            result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 上传成功！")
                print(f"🎉 模型地址: https://modelscope.cn/{username}/{model_name}")
                return True
            else:
                print(f"❌ 推送失败: {result.stderr}")
                
                # 如果main分支失败，尝试master分支
                print("🔄 尝试推送到master分支...")
                result2 = subprocess.run(['git', 'push', '-u', 'origin', 'master'], 
                                       capture_output=True, text=True)
                
                if result2.returncode == 0:
                    print("✅ 推送到master分支成功！")
                    return True
                else:
                    print(f"❌ 推送到master分支也失败: {result2.stderr}")
                    return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Git操作失败: {e}")
            return False
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 ModelScope Git认证修复工具")
    print("=" * 50)
    
    if len(sys.argv) < 4:
        print("使用方法:")
        print("  python fix_git_auth_upload.py <本地路径> <用户名> <模型名> [token]")
        print("")
        print("示例:")
        print("  python fix_git_auth_upload.py /root/flux_model_cache/flux-dev xiaosa FLUX.1-dev")
        print("  python fix_git_auth_upload.py /root/flux_model_cache/flux-dev xiaosa FLUX.1-dev your_token")
        return
    
    local_path = sys.argv[1]
    username = sys.argv[2]
    model_name = sys.argv[3]
    token = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('MODELSCOPE_API_TOKEN')
    
    if not token:
        print("❌ 需要ModelScope Token")
        print("请设置环境变量或作为参数提供:")
        print("  export MODELSCOPE_API_TOKEN='your_token'")
        print("  或")
        print("  python fix_git_auth_upload.py <路径> <用户名> <模型名> <token>")
        return
    
    if not os.path.exists(local_path):
        print(f"❌ 本地路径不存在: {local_path}")
        return
    
    print(f"📁 本地模型: {local_path}")
    print(f"👤 用户名: {username}")
    print(f"📦 模型名: {model_name}")
    print(f"🔑 Token: {token[:10]}...")
    
    print("\n💡 注意事项:")
    print("1. 请确保在ModelScope网页上已创建仓库")
    print(f"2. 仓库地址: https://modelscope.cn/{username}/{model_name}")
    print("3. 如果仓库不存在，请先在网页上创建")
    
    # 询问是否继续
    response = input("\n继续上传？(y/n): ").strip().lower()
    if response != 'y':
        print("❌ 用户取消操作")
        return
    
    # 执行上传
    if create_and_upload_to_modelscope(local_path, username, model_name, token):
        print("\n🎉 上传完成！")
    else:
        print("\n❌ 上传失败")
        print("\n💡 建议:")
        print("1. 检查网络连接")
        print("2. 验证Token有效性")
        print("3. 确认仓库已在ModelScope网页上创建")
        print("4. 检查用户名和模型名拼写")

if __name__ == "__main__":
    main()