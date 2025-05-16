import json
import os
import argparse
from glob import glob

def merge_json_lists(input_files, output_file):
    """
    JSONJSON
    
    Args:
        input_files (list): JSON
        output_file (str): 
    """
    merged_data = []
    total_files = len(input_files)
    
    print(f" {total_files} JSON...")
    
    for i, file_path in enumerate(input_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 
                if isinstance(data, list):
                    merged_data.extend(data)
                    print(f"[{i}/{total_files}] : {file_path} ( {len(data)} )")
                else:
                    print(f"[{i}/{total_files}] : {file_path} ")
        except json.JSONDecodeError:
            print(f"[{i}/{total_files}] : {file_path} JSON")
        except Exception as e:
            print(f"[{i}/{total_files}]  {file_path} : {str(e)}")
    
    # 
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"!  {total_files}  {len(merged_data)} ")
    print(f": {output_file}")

def main():
    # python merge_json.py file1.json file2.json file3.json -o merged_output.json

    parser = argparse.ArgumentParser(description="JSON")
    parser.add_argument("input", nargs='+', help=" ( *.json)")
    parser.add_argument("-o", "--output", default="merged.json", help=" (: merged.json)")
    
    args = parser.parse_args()
    
    # 
    input_files = []
    for path in args.input:
        if '*' in path or '?' in path:
            input_files.extend(glob(path))
        else:
            input_files.append(path)
    
    # 
    if not input_files:
        print(": ")
        return
    
    merge_json_lists(input_files, args.output)

if __name__ == "__main__":
    main()