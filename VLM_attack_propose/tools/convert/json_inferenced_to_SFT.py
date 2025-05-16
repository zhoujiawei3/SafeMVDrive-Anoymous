import json
import os
import argparse

def convert_json_format(input_file, output_file):
    # JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 
    converted_data = []
    
    # 
    for item in original_data:
        # 
        new_format = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image> {item['question']}"
                },
                {
                    "role": "assistant",
                    "content": item['full_response']
                }
            ],
            "images": [
                item['image_path']
            ]
        }
        
        # 
        converted_data.append(new_format)
    
    # JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f" {output_file}")
    print(f" {len(converted_data)} ")

def parse_arguments():
    parser = argparse.ArgumentParser(description='JSON')
    parser.add_argument('--input', '-i', type=str, required=True, help='JSON')
    parser.add_argument('--output', '-o', type=str, required=True, help='JSON')
    return parser.parse_args()

if __name__ == "__main__":
    # 
    args = parse_arguments()
    
    # 
    convert_json_format(args.input, args.output)
