#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版ModelScope上传脚本
解决namespace和Repository API问题
"""

import os
import sys
import shutil
import tempfile

def check_modelscope_user():
    """检查ModelScope当前用户信息"""
    try:
        print("🔍 检查ModelScope用户信息...")
        
        from modelscope.hub.api import HubApi
        api = HubApi()
        
        # 尝试获取用户信息
        try:
            # 方法1: 尝试通过环境变量验证token
            token = os.environ.get('MODELSCOPE_API_TOKEN')
            if not token:
                print("❌ 未设置MODELSCOPE_API_TOKEN环境变量")
                return None
            
            print("✅ 找到ModelScope token")
            print("💡 无法通过API自动获取用户名")
            return None
            
        except Exception as e:
            print(f"❌ Token检查失败: {e}")
            return None
            
    except ImportError:
        print("❌ ModelScope库未安装")
        return None
    except Exception as e:
        print(f"❌ 检查用户信息失败: {e}")
        return None

def list_user_models():
    """列出用户的现有模型"""
    print("\n📋 由于API限制，无法自动列出现有模型")
    print("💡 请手动确认您的ModelScope用户名:")
    print("   1. 访问: https://modelscope.cn/profile")
    print("   2. 查看您的个人资料页面")
    print("   3. 找到您的用户名或ID")
    return None

def suggest_valid_usernames():
    """建议有效的用户名格式"""
    print("\n📝 ModelScope用户名要求:")
    print("• 只能包含字母、数字、连字符(-)")
    print("• 不能以连字符开头或结尾")
    print("• 长度通常为3-20个字符")
    print("• 常见格式: username、user-name、user123")
    
def check_namespace_exists(namespace):
    """检查namespace是否存在"""
    print(f"\n🔍 检查namespace '{namespace}'...")
    print("💡 由于API限制，建议手动验证用户名:")
    print(f"   访问: https://modelscope.cn/{namespace}")
    print("   如果页面存在，则用户名有效")
    
    # 简单的用户名格式验证
    import re
    if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$', namespace) or re.match(r'^[a-zA-Z0-9]+$', namespace):
        print(f"✅ '{namespace}' 格式符合要求")
        return True
    else:
        print(f"❌ '{namespace}' 格式不符合要求")
        return False

def upload_to_modelscope_fixed(local_path, ms_model_name="test-user/FLUX.1-dev"):
    """修复版ModelScope上传函数"""
    try:
        print(f"📤 开始上传到ModelScope: {ms_model_name}")
        
        # 分离用户名和模型名
        if "/" in ms_model_name:
            namespace, model_name = ms_model_name.split("/", 1)
        else:
            print("❌ 模型名格式错误，应为 'namespace/model-name'")
            return False
        
        print(f"👤 Namespace: {namespace}")
        print(f"📦 Model: {model_name}")
        
        # 检查环境变量
        if not os.environ.get('MODELSCOPE_API_TOKEN'):
            print("❌ 未设置MODELSCOPE_API_TOKEN环境变量")
            return False
        
        from modelscope.hub.api import HubApi
        api = HubApi()
        
        # 方法1: 尝试直接创建仓库（使用更简单的参数）
        try:
            print("🔨 尝试创建ModelScope仓库...")
            result = api.create_model(
                model_id=ms_model_name,
                visibility=1,
                license='apache-2.0'  # 使用更通用的许可证
            )
            print(f"✅ 仓库创建成功")
        except Exception as e:
            error_msg = str(e)
            if "namespace" in error_msg.lower():
                print(f"❌ Namespace问题: {error_msg}")
                print("💡 建议:")
                print("  1. 检查您的ModelScope用户名是否正确")
                print("  2. 尝试使用其他namespace")
                print("  3. 或创建新的组织")
                return False
            else:
                print(f"ℹ️  仓库可能已存在: {e}")
        
        # 方法2: 使用git命令手动上传（避免Repository API问题）
        print("📥 使用git方式上传...")
        
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as temp_work_dir:
            # 克隆仓库
            repo_url = f"https://www.modelscope.cn/{ms_model_name}.git"
            clone_dir = os.path.join(temp_work_dir, "repo")
            
            print(f"📥 克隆仓库: {repo_url}")
            result = os.system(f"git clone {repo_url} {clone_dir}")
            
            if result != 0:
                print("❌ 克隆失败，可能仓库不存在或无权限")
                return False
            
            # 复制文件
            print("📝 复制模型文件...")
            for item in os.listdir(local_path):
                src = os.path.join(local_path, item)
                dst = os.path.join(clone_dir, item)
                
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            # Git操作
            os.chdir(clone_dir)
            
            print("📋 检查文件变更...")
            result = os.system("git add .")
            if result != 0:
                print("❌ git add 失败")
                return False
            
            # 检查是否有变更
            result = os.system("git diff --cached --quiet")
            if result == 0:
                print("ℹ️  没有文件变更")
                return True
            
            print("💾 提交变更...")
            commit_msg = "Upload FLUX.1-dev model from Hugging Face"
            result = os.system(f'git commit -m "{commit_msg}"')
            if result != 0:
                print("❌ git commit 失败")
                return False
            
            print("🚀 推送到ModelScope...")
            result = os.system("git push origin main")
            if result != 0:
                result = os.system("git push origin master")  # 尝试master分支
            
            if result != 0:
                print("❌ git push 失败")
                return False
            
            print(f"🎉 上传成功！")
            print(f"🔗 ModelScope链接: https://modelscope.cn/models/{ms_model_name}")
            return True
            
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        print("\n💡 建议:")
        print("1. 检查网络连接")
        print("2. 确认ModelScope token有效")
        print("3. 检查用户名和权限")
        print("4. 尝试在ModelScope网页上手动创建仓库")
        return False

def main():
    """主函数"""
    print("🔧 ModelScope上传修复工具")
    print("=" * 40)
    
    # 检查参数
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python fix_modelscope_upload.py <本地模型路径> [模型名称]")
        print("")
        print("示例:")
        print("  python fix_modelscope_upload.py ~/flux_model_cache/flux-dev")
        print("  python fix_modelscope_upload.py ~/flux_model_cache/flux-dev username/FLUX.1-dev")
        print("  python fix_modelscope_upload.py ~/flux_model_cache/flux-dev --check-user")
        return
    
    # 如果是检查用户模式
    if len(sys.argv) == 2 and sys.argv[1] == "--check-user":
        print("🔍 用户信息检查模式")
        print("-" * 30)
        
        # 检查当前用户
        current_user = check_modelscope_user()
        
        # 列出现有模型
        inferred_user = list_user_models()
        
        # 建议用户名格式
        suggest_valid_usernames()
        
        if current_user:
            print(f"\n✅ 建议使用: {current_user}/FLUX.1-dev")
        elif inferred_user:
            print(f"\n✅ 建议使用: {inferred_user}/FLUX.1-dev")
        else:
            print("\n❌ 无法自动确定用户名，请手动指定")
        return
    
    local_path = sys.argv[1]
    ms_model_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(local_path):
        print(f"❌ 本地路径不存在: {local_path}")
        return
    
    print(f"📁 本地模型: {local_path}")
    
    # 如果没有指定模型名，尝试自动确定
    if not ms_model_name:
        print("\n🔍 自动检查用户信息...")
        
        # 尝试获取用户名
        username = check_modelscope_user()
        if not username:
            username = list_user_models()
        
        if username:
            ms_model_name = f"{username}/FLUX.1-dev"
            print(f"✅ 自动确定模型名: {ms_model_name}")
        else:
            print("❌ 无法自动确定用户名")
            suggest_valid_usernames()
            print("\n请手动指定模型名:")
            print("  python fix_modelscope_upload.py <路径> your-username/FLUX.1-dev")
            return
    
    # 验证namespace
    if "/" in ms_model_name:
        namespace = ms_model_name.split("/")[0]
        if not check_namespace_exists(namespace):
            print(f"\n❌ Namespace '{namespace}' 验证失败")
            print("💡 建议:")
            print("1. 检查用户名拼写")
            print("2. 确认您有权限使用此namespace")
            print("3. 在ModelScope网页上手动创建仓库")
            return
    
    print(f"🎯 目标仓库: {ms_model_name}")
    
    # 执行上传
    if upload_to_modelscope_fixed(local_path, ms_model_name):
        print("\n🎉 修复版上传完成！")
    else:
        print("\n❌ 上传失败")

if __name__ == "__main__":
    main()
