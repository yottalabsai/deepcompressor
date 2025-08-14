# ModelScope 认证指南

## 概述

ModelScope认证采用与Hugging Face类似的方式，用户需要先登录后才能下载私有模型或获得更好的下载体验。

## 认证方式

### 方式1: 命令行登录（推荐）

```bash
# 首先安装ModelScope
pip install modelscope

# 交互式登录（系统会提示输入token）
modelscope login

# 或者直接传入token（非交互式）
modelscope login --token "your_token_here"
```

推荐在自动化脚本中使用 `--token` 参数，在手动操作时使用交互式登录。

### 方式2: 环境变量

```bash
# 设置环境变量
export MODELSCOPE_API_TOKEN="your_token_here"

# 永久设置（添加到shell配置文件）
echo 'export MODELSCOPE_API_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### 方式3: Python API登录

```python
from modelscope.hub.api import HubApi

# 登录
api = HubApi()
api.login("your_token_here")

# 检查登录状态
print("登录成功！" if api.get_token() else "未登录")
```

### 使用场景对比

| 场景 | 推荐方式 | 命令 |
|-----|---------|------|
| 手动操作 | 交互式登录 | `modelscope login` |
| 自动化脚本 | token参数 | `modelscope login --token "$TOKEN"` |
| 环境变量 | 环境变量 | `export MODELSCOPE_API_TOKEN="token"` |
| Python代码 | API登录 | `HubApi().login("token")` |

## 获取Token

1. 访问 [ModelScope官网](https://modelscope.cn)
2. 注册并登录账户
3. 进入个人中心 → API Token
4. 生成新的token或查看现有token
5. 复制token用于登录

## 在DeepCompressor中使用

### 对于公开模型

大多数公开模型不需要认证即可下载：

```yaml
model:
  name: qwen-7b-chat
  use_modelscope: true
  path: "qwen/Qwen-7B-Chat"  # 公开模型
```

### 对于私有模型

私有模型需要先登录：

```bash
# 1. 先登录（交互式）
modelscope login

# 或者使用token登录（自动化）
modelscope login --token "your_token_here"

# 2. 然后使用私有模型
```

```yaml
model:
  name: my-private-model
  use_modelscope: true
  path: "your-org/private-model-name"
```

## 常见问题

### Q: 如何检查是否已登录？

```python
from modelscope.hub.api import HubApi
api = HubApi()
token = api.get_token()
if token:
    print(f"已登录，token: {token[:10]}...")
else:
    print("未登录")
```

### Q: 如何退出登录？

```bash
modelscope logout
```

或删除token文件：
```bash
rm ~/.modelscope/token
```

### Q: 登录信息存储在哪里？

ModelScope将token存储在用户主目录下：
- Linux/Mac: `~/.modelscope/token`
- Windows: `%USERPROFILE%\.modelscope\token`

### Q: 遇到认证错误怎么办？

常见错误及解决方案：

1. **Token无效**
   ```
   Error: Invalid token
   ```
   解决：重新生成token并登录

2. **网络问题**
   ```
   Error: Connection timeout
   ```
   解决：检查网络连接，可能需要配置代理

3. **权限不足**
   ```
   Error: Access denied
   ```
   解决：确认账户有权限访问该模型

## 最佳实践

1. **使用环境变量**：在生产环境中使用环境变量而不是硬编码token
2. **定期更新token**：定期刷新token以确保安全性
3. **权限最小化**：只申请必要的权限
4. **安全存储**：不要在代码或配置文件中明文存储token

## 示例脚本

检查ModelScope认证状态的脚本：

```python
#!/usr/bin/env python3
"""检查ModelScope认证状态"""

def check_modelscope_auth():
    try:
        from modelscope.hub.api import HubApi
        
        api = HubApi()
        token = api.get_token()
        
        if token:
            print("✅ ModelScope已登录")
            print(f"Token: {token[:10]}...")
            return True
        else:
            print("❌ ModelScope未登录")
            print("请运行: modelscope login")
            return False
            
    except ImportError:
        print("❌ ModelScope未安装")
        print("请运行: pip install modelscope")
        return False
    except Exception as e:
        print(f"❌ 检查认证状态时出错: {e}")
        return False

if __name__ == "__main__":
    check_modelscope_auth()
```

## 总结

ModelScope的认证机制类似于Hugging Face，用户需要：

1. 先注册ModelScope账户
2. 获取API Token
3. 使用 `modelscope login` 命令登录
4. 然后就可以正常使用DeepCompressor的ModelScope功能

这种方式简单、安全，用户完全控制自己的认证信息。