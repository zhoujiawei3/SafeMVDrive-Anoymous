import os
import sys

def remove_prefix_from_files(directory_path):
    """
    'example_bev_100_val_random2_bug_fix_after6frames'
    
    
    Args:
        directory_path (str): 
    """
    # 
    prefix = "example_bev_200_val_random2_bug_fix_after6frames"
    
    # 
    if not os.path.isdir(directory_path):
        print(f": '{directory_path}' ")
        return
    
    # 
    renamed_count = 0
    skipped_count = 0
    
    # 
    for filename in os.listdir(directory_path):
        # 
        old_path = os.path.join(directory_path, filename)
        
        # 
        if os.path.isdir(old_path):
            continue
        
        # 
        if filename.startswith(prefix):
            #  - 
            new_filename = filename[len(prefix):]
            
            # 
            if new_filename.startswith("_"):
                new_filename = new_filename[1:]
            
            # 
            new_path = os.path.join(directory_path, new_filename)
            
            try:
                # 
                os.rename(old_path, new_path)
                print(f": '{filename}' -> '{new_filename}'")
                renamed_count += 1
            except Exception as e:
                print(f" '{filename}' : {e}")
        else:
            print(f": '{filename}' ()")
            skipped_count += 1
    
    print(f"\n!  {renamed_count}  {skipped_count} ")
    
    if renamed_count == 0:
        print(" 'example_bev_100_val_random2_bug_fix_after6frames' ")

if __name__ == "__main__":
    # 
    
    directory="/home//VLM_attack_propose/example_bev_200_val_random2_bug_fix_after6frames"
    # 
    directory = directory.strip('"\'')
    
    # 
    remove_prefix_from_files(directory)