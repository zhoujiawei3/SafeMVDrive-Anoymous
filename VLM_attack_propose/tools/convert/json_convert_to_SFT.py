import json
import os
from tqdm import tqdm

def convert_to_sharegpt_format(input_file, output_file, image_root_dir):
    """
    Convert the original data format to ShareGPT format.
    
    Args:
        input_file: Path to input JSON file.
        output_file: Path to output JSON file.
        image_root_dir: Root directory for images.
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = []
    
    # Process each item in the data
    if isinstance(data, list):
        items = data
    else:
        items = [data]
        
    for item in tqdm(items, desc="Processing items"):
        # Handle the assistant response
        if len(item["conversations"]) > 1:
            assistant_value = item["conversations"][1]["value"]
            
            if isinstance(assistant_value, dict):
                if assistant_value:  # If dictionary is not empty
                    # For each key in the dictionary, create a separate item
                    for key in assistant_value.keys():
                        new_item = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "<image>"+item["conversations"][0]["value"]
                                },
                                {
                                    "role": "assistant",
                                    "content": key
                                }
                            ],
                            "images": [os.path.join(image_root_dir, item["image"])]
                        }
                        output_data.append(new_item)
                else:
                    # If dictionary is empty
                    new_item = {
                        "messages": [
                            {
                                "role": "user",
                                "content": "<image>"+item["conversations"][0]["value"]
                            },
                            {
                                "role": "assistant",
                                "content": "no vehicle is appropriate"
                            }
                        ],
                        "images": [os.path.join(image_root_dir, item["image"])]
                    }
                    output_data.append(new_item)
            else:
                # If not a dictionary, process as before
                new_item = {
                    "messages": [
                        {
                            "role": "user",
                            "content": "<image>"+item["conversations"][0]["value"]
                        },
                        {
                            "role": "assistant",
                            "content": assistant_value
                        }
                    ],
                    "images": [os.path.join(image_root_dir, item["image"])]
                }
                output_data.append(new_item)
        else:
            # If there's no assistant response, just add the user message
            new_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>"+item["conversations"][0]["value"]
                    }
                ],
                "images": [os.path.join(image_root_dir, item["image"])]
            }
            output_data.append(new_item)
    
    # Save the output directly
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(output_data)} items. Output saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Configuration
    input_file = "/home//VLM_attack_propose/annotation/mini-data_new_1500_continue_train_after6frames_random7_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_only_no_type.json"  # Your input JSON file
    output_file = "/home//VLM_attack_propose/annotation/mini-data_new_1500_continue_train_after6frames_random7_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_only_no_type_for_SFT.json"
    image_root_dir = "/home//VLM_attack_propose/example_rgb_1500_continue_train_after6frames_random7_bug_fix"
    
    # Run the conversion
    convert_to_sharegpt_format(input_file, output_file, image_root_dir)