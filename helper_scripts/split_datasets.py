import splitfolders

# Define paths to your datasets
dataset_A_path = 'E:\\DL\\test'
dataset_B_path = 'E:\\DL\\pure_target_dataset'

# Define the output directories for the splits
output_A_path = './datasets/attempt_one/A'
output_B_path = './datasets/attempt_one/B'

# Split with a ratio (e.g., 80% training, 20% testing)
split_ratio = (0.8, 0.2)  # Adjust the ratio as needed

# Split dataset A
splitfolders.ratio(dataset_A_path, output=output_A_path, seed=1337, ratio=split_ratio)

# Split dataset B
splitfolders.ratio(dataset_B_path, output=output_B_path, seed=1337, ratio=split_ratio)
