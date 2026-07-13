import json
from collections import defaultdict

def extract_keys(data, parent_key='', separator='.', keys=None):
    if keys is None:
        keys = []
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{parent_key}{separator}{key}" if parent_key else key
            keys.append(current_path)
            extract_keys(value, current_path, separator, keys)
    elif isinstance(data, list):
        for item in data:
            extract_keys(item, parent_key, separator, keys)
    return keys

# 从json.json文件读取数据
with open('json.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取并去重键
all_keys = extract_keys(data)
unique_keys = list(dict.fromkeys(all_keys))  # 保留首次出现的顺序


def group_and_sort_keys(all_keys):
    # 初始化数据结构
    groups = defaultdict(list)
    seen = defaultdict(set)
    all_groups = set()

    # 遍历所有键进行分类
    for key in all_keys:
        parts = key.split('.')
        if not parts: continue
        
        main_key = parts[0]
        all_groups.add(main_key)
        
        if len(parts) > 1:
            sub_key = parts[-1]
            if sub_key not in seen[main_key]:
                seen[main_key].add(sub_key)
                groups[main_key].append(sub_key)

    # 补全没有子键的一级键
    for key in all_groups:
        if key not in groups:
            groups[key] = []

    # 按字母顺序排序
    sorted_groups = sorted(groups.items(), key=lambda x: x[0].lower())
    final = []
    for main_key, sub_keys in sorted_groups:
        final.append((
            main_key,
            sorted(sub_keys, key=lambda x: x.lower())  # 子键字母排序
        ))
    
    return final



result = group_and_sort_keys(unique_keys)

# 打印结果
print("所有键及其层级路径:")
for key in result:
    print(key)
