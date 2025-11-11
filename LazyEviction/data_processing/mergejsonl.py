import json
from pathlib import Path

def merge_jsonl_files(file1_path, file2_path, output_path):
    """
    合并两个JSONL文件，处理可能的ID冲突
    
    参数:
        file1_path: 第一个JSONL文件路径
        file2_path: 第二个JSONL文件路径
        output_path: 合并后的输出路径
    """
    seen_ids = set()
    merged_data = []
    
    def load_file(file_path):
        nonlocal seen_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    original_id = data.get('id', '')
                    
                    # 处理ID冲突
                    new_id = original_id
                    suffix = 1
                    while new_id in seen_ids:
                        new_id = f"{original_id}_{suffix}"
                        suffix += 1
                    
                    if new_id != original_id:
                        data['id'] = new_id
                    
                    seen_ids.add(new_id)
                    merged_data.append(data)
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line.strip()}")
    
    # 加载两个文件
    print(f"正在加载 {file1_path}...")
    load_file(file1_path)
    
    print(f"正在加载 {file2_path}...")
    load_file(file2_path)
    
    # 写入合并后的文件
    print(f"正在写入合并结果到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for item in merged_data:
            out_file.write(json.dumps(item) + '\n')
    
    print(f"合并完成！总记录数: {len(merged_data)}")
    print(f"原始文件1记录数: {len([d for d in merged_data if d.get('id', '').startswith(Path(file1_path).stem)])}")
    print(f"原始文件2记录数: {len([d for d in merged_data if d.get('id', '').startswith(Path(file2_path).stem)])}")

if __name__ == "__main__":
    # 使用示例
    file1 = "/root/code/LongReasoning_zhl/datasets/aime/AIME_Dataset_1983_2023.jsonl"
    file2 = "/root/code/LongReasoning_zhl/datasets/aime/aime2024-2.jsonl"
    output = "/root/code/LongReasoning_zhl/datasets/aime/AIME_Dataset_1983_2024.jsonl"
    
    merge_jsonl_files(file1, file2, output)