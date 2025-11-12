#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量为所有图像处理节点添加 seed 参数"""

import re

# 读取文件
with open('oaigc.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 需要添加 seed 的节点类名列表（剩余的24个）
nodes_to_update = [
    'OAIZhengJianZhao', 'OAIRenwuchangjingronghe', 'OAIZitaiqianyi', 'OAIQianzibaitai',
    'OAIQwenEdit', 'OAIFantuichutu', 'OAIXiangaochengtu', 'OAIChongdaguang',
    'OAIZhenrenshouban', 'OAIQushuiyin', 'OAILaozhaopian', 'OAIDianshang',
    'OAIFlux', 'OAIKeling', 'OAIJipuli', 'OAIChanpin',
    'OAIGaoqing', 'OAIMaopei', 'OAIWanwuqianyi', 'OAIKuotu',
    'OAIKoutu', 'OAIJianzhi', 'OAIShangse', 'OAIHuanyi',
    'OAIWanwuhuanbeijing', 'OAIImagePromptReverse'
]

seed_param = '''                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),'''

modified_count = 0

for node_name in nodes_to_update:
    # 模式1: required 部分末尾添加 seed（在最后一个参数后）
    pattern1 = rf'(class {node_name}:.*?def INPUT_TYPES\(cls\):.*?"required": \{{.*?)\n(\s+)(\}})\n(\s+\}})\n(\s+RETURN_TYPES)'
    
    def add_seed_to_input(match):
        before = match.group(1)
        indent1 = match.group(2)
        close_brace1 = match.group(3)
        indent2 = match.group(4)
        close_brace2 = match.group(5)
        return_types = match.group(6)
        
        # 在 required 的最后一个元素后添加 seed
        new_text = f"{before}\n{seed_param}\n{indent1}{close_brace1}\n{indent2}{close_brace2}\n{indent2}{return_types}"
        return new_text
    
    # 使用 DOTALL 模式匹配跨行内容
    content, count = re.subn(pattern1, add_seed_to_input, content, count=1, flags=re.DOTALL)
    
    if count > 0:
        print(f"✓ 为 {node_name} 的 INPUT_TYPES 添加了 seed 参数")
        modified_count += count
    
    # 模式2: 修改 process_image 或 reverse_prompt 函数签名
    # 查找函数定义并在最后一个参数后添加 seed
    pattern2 = rf'(class {node_name}:.*?def (?:process_image|reverse_prompt)\(self,[^)]+)(\):)'
    
    def add_seed_to_function(match):
        func_def = match.group(1)
        close_paren = match.group(2)
        
        # 如果已经有 seed 参数，跳过
        if ', seed' in func_def or 'seed=' in func_def:
            return match.group(0)
        
        # 添加 seed 参数
        return f"{func_def}, seed{close_paren}"
    
    content, count = re.subn(pattern2, add_seed_to_function, content, count=1, flags=re.DOTALL)
    
    if count > 0:
        print(f"✓ 为 {node_name} 的函数签名添加了 seed 参数")
        modified_count += count

# 写回文件
with open('oaigc.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n总共修改了 {modified_count} 处")
print("✅ 批量添加 seed 参数完成！")
