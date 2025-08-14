#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModelScope用户名检查工具
"""

import re
import requests
import os

def check_username_format(username):
    """检查用户名格式"""
    print(f"🔍 检查用户名格式: {username}")
    
    # 基本格式检查
    if not username:
        print("❌ 用户名不能为空")
        return False
    
    if len(username) < 3 or len(username) > 39:
        print("❌ 用户名长度应在3-39个字符之间")
        return False
    
    # 字符检查
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$', username) and not re.match(r'^[a-zA-Z0-9]+$', username):
        print("❌ 用户名格式不正确")
        print("   • 只能包含字母、数字、连字符(-)")
        print("   • 不能以连字符开头或结尾")
        return False
    
    print("✅ 用户名格式正确")
    return True

def check_username_exists(username):
    """检查用户名是否存在"""
    print(f"\n🌐 检查用户名是否存在: {username}")
    
    try:
        # 尝试访问用户页面
        url = f"https://modelscope.cn/{username}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ 用户名 '{username}' 存在")
            print(f"🔗 用户页面: {url}")
            return True
        elif response.status_code == 404:
            print(f"❌ 用户名 '{username}' 不存在")
            return False
        else:
            print(f"❓ 无法确定，HTTP状态码: {response.status_code}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ 网络检查失败: {e}")
        return None

def suggest_alternatives(username):
    """建议可能的用户名"""
    print(f"\n💡 可能的用户名建议:")
    
    alternatives = [
        username.lower(),
        username.replace('_', '-'),
        username.replace('-', '_'),
        f"{username}123",
        f"ms-{username}",
        f"{username}-ms"
    ]
    
    for i, alt in enumerate(alternatives, 1):
        if alt != username and check_username_format(alt):
            print(f"  {i}. {alt}")

def check_token_validity():
    """检查ModelScope token有效性"""
    print("\n🔑 检查ModelScope Token...")
    
    token = os.environ.get('MODELSCOPE_API_TOKEN')
    if not token:
        print("❌ 未设置MODELSCOPE_API_TOKEN环境变量")
        print("💡 设置方法:")
        print("   export MODELSCOPE_API_TOKEN='your_token_here'")
        return False
    
    print(f"✅ 找到token: {token[:10]}...")
    
    # 简单的token格式检查
    if len(token) < 20:
        print("⚠️  Token长度可能不正确（通常应该较长）")
    
    return True

def main():
    """主函数"""
    print("🔍 ModelScope用户名检查工具")
    print("=" * 40)
    
    # 检查token
    check_token_validity()
    
    print("\n" + "=" * 40)
    
    # 交互式检查用户名
    while True:
        username = input("\n请输入要检查的用户名 (或输入 'q' 退出): ").strip()
        
        if username.lower() == 'q':
            print("👋 再见！")
            break
        
        if not username:
            print("请输入有效的用户名")
            continue
        
        print("-" * 30)
        
        # 格式检查
        format_ok = check_username_format(username)
        
        if format_ok:
            # 存在性检查
            exists = check_username_exists(username)
            
            if exists:
                print(f"\n🎉 '{username}' 可以使用！")
                print(f"📝 建议的模型名: {username}/FLUX.1-dev")
            elif exists is False:
                print(f"\n❌ '{username}' 不存在")
                suggest_alternatives(username)
            else:
                print(f"\n❓ 无法确认 '{username}' 的状态，建议手动检查")
        else:
            suggest_alternatives(username)
        
        print("\n" + "=" * 40)

if __name__ == "__main__":
    main()