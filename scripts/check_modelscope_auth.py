#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelScope认证状态检查脚本

使用方法:
python scripts/check_modelscope_auth.py
"""

import os
import sys

def check_modelscope_installation():
    """检查ModelScope是否已安装"""
    try:
        import modelscope
        print(f"✅ ModelScope已安装，版本: {modelscope.__version__}")
        return True
    except ImportError:
        print("❌ ModelScope未安装")
        print("   安装命令: pip install modelscope")
        return False

def check_modelscope_auth():
    """检查ModelScope认证状态"""
    try:
        from modelscope.hub.api import HubApi
        
        api = HubApi()
        token = api.get_token()
        
        if token:
            print("✅ ModelScope已认证")
            print(f"   Token: {token[:10]}...")
            return True
        else:
            print("⚠️  ModelScope未认证")
            print("   这不影响使用公开模型，但私有模型需要认证")
            return False
            
    except Exception as e:
        print(f"❌ 检查认证状态时出错: {e}")
        return False

def check_environment_variables():
    """检查环境变量"""
    env_vars = ['MODELSCOPE_API_TOKEN', 'MODELSCOPE_TOKEN']
    found_vars = []
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            found_vars.append(var)
            print(f"✅ 找到环境变量 {var}: {value[:10]}...")
    
    if not found_vars:
        print("ℹ️  未找到ModelScope相关环境变量")
        print("   这是正常的，可以使用 'modelscope login' 命令登录")
    
    return len(found_vars) > 0

def show_login_instructions():
    """显示登录说明"""
    print("\n📋 ModelScope登录方法:")
    print("   1. 命令行登录: modelscope login")
    print("   2. 环境变量: export MODELSCOPE_API_TOKEN='your_token'")
    print("   3. 获取Token: 访问 https://modelscope.cn -> 个人中心 -> API Token")

def test_model_access():
    """测试模型访问（可选）"""
    print("\n🧪 测试ModelScope模型访问...")
    
    try:
        from modelscope import snapshot_download
        
        # 尝试获取一个公开模型的信息（不实际下载）
        test_model = "damo/nlp_structbert_word-segmentation_chinese-base"
        
        print(f"   测试模型: {test_model}")
        print("   注意: 这只是测试连接，不会下载模型")
        
        # 这里只是检查能否连接到ModelScope，不实际下载
        try:
            # 模拟一个快速的连接测试
            import requests
            response = requests.get("https://modelscope.cn", timeout=5)
            if response.status_code == 200:
                print("✅ 可以连接到ModelScope服务器")
            else:
                print(f"⚠️  连接到ModelScope服务器响应异常: {response.status_code}")
        except Exception as e:
            print(f"⚠️  连接到ModelScope服务器失败: {e}")
            
    except ImportError:
        print("   跳过模型访问测试（ModelScope未安装）")
    except Exception as e:
        print(f"   模型访问测试失败: {e}")

def main():
    """主函数"""
    print("🔍 ModelScope认证状态检查\n")
    
    # 检查安装
    installed = check_modelscope_installation()
    
    if not installed:
        print("\n❌ 请先安装ModelScope库")
        return
    
    print()
    
    # 检查认证状态
    auth_by_cli = check_modelscope_auth()
    
    print()
    
    # 检查环境变量
    auth_by_env = check_environment_variables()
    
    print()
    
    # 总结
    if auth_by_cli or auth_by_env:
        print("🎉 ModelScope认证配置正常！")
        print("   可以下载公开和私有模型")
    else:
        print("ℹ️  ModelScope未认证")
        print("   可以下载公开模型，私有模型需要先认证")
        show_login_instructions()
    
    # 测试连接
    test_model_access()
    
    print("\n📚 更多信息:")
    print("   - ModelScope认证指南: docs/ModelScope-Authentication.md")
    print("   - ModelScope支持文档: docs/ModelScope-Support.md")

if __name__ == "__main__":
    main()