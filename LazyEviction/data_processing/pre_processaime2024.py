import json

def transform_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            
            # 构建新的JSON对象
            transformed = {
                "question": data["question"],
                "answer": data["answer"],
                "cot": data.get("cot", ""),  # 使用已有cot或空字符串
                "id": data["id"]
            }
            
            # 写入新文件
            outfile.write(json.dumps(transformed) + '\n')

if __name__ == "__main__":
    input_jsonl = "/root/code/LongReasoning_zhl/datasets/aime/aime2024.jsonl"  # 原始JSONL文件路径
    output_jsonl = "/root/code/LongReasoning_zhl/datasets/aime/aime2024-2.jsonl"  # 转换后的JSONL文件路径
    
    transform_jsonl(input_jsonl, output_jsonl)
    print(f"转换完成，结果已保存到 {output_jsonl}")