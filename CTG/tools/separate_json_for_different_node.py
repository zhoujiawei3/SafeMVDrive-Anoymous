import json
import os
import argparse
from collections import defaultdict

def split_json_file(json_path, num_parts):
    """
    Split a JSON file into multiple parts using a greedy algorithm to balance 'reward' elements.
    
    Args:
        json_path (str): Path to the input JSON file
        num_parts (int): Number of parts to split the file into
    """
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Group data by sample_token
    token_groups = defaultdict(list)
    for item in data:
        token = item.get('sample_token', '')
        token_groups[token].append(item)
    
    # Count reward elements in each group
    group_reward_counts = {}
    for token, items in token_groups.items():
        group_reward_counts[token] = sum(len(item.get('reward', {})) for item in items)
    
    # Sort groups by reward count (descending)
    sorted_groups = sorted(group_reward_counts.keys(), key=lambda x: group_reward_counts[x], reverse=True)
    
    # Initialize parts and their reward counts
    parts = [[] for _ in range(num_parts)]
    current_reward_counts = [0] * num_parts
    
    # Use greedy algorithm to distribute groups
    for token in sorted_groups:
        # Find the part with the minimum reward count
        min_count_part = current_reward_counts.index(min(current_reward_counts))
        
        # Add the entire group to that part
        items = token_groups[token]
        parts[min_count_part].extend(items)
        current_reward_counts[min_count_part] += group_reward_counts[token]
    
    # Calculate total reward elements
    total_reward_elements = sum(group_reward_counts.values())
    print(f"Total data records: {len(data)}, with {len(token_groups)} different sample_token groups")
    print(f"Total reward elements: {total_reward_elements}")
    print(f"Splitting into {num_parts} parts")
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_dir = os.path.join(os.path.dirname(json_path), base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each part to a separate file
    for i, part_data in enumerate(parts):
        part_file = os.path.join(output_dir, f"part_{i}.json")
        
        with open(part_file, "w") as f:
            json.dump(part_data, f, indent=2)
        
        # Calculate statistics for this part
        part_reward_count = sum(len(item.get("reward", {})) for item in part_data)
        unique_tokens = len(set(item.get("sample_token", "") for item in part_data))
        
        print(f"Wrote {len(part_data)} records to {part_file}")
        print(f"  - Contains {unique_tokens} unique sample_tokens")
        print(f"  - Contains {part_reward_count} reward elements ({part_reward_count/total_reward_elements*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Split a JSON file into multiple parts")
    parser.add_argument("--json_path", help="Path to the input JSON file")
    parser.add_argument("--num_parts", type=int, default=3, help="Number of parts to split into (default: 5)")
    
    args = parser.parse_args()
    
    split_json_file(args.json_path, args.num_parts)

if __name__ == "__main__":
    main()