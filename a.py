import pickle
import glob
import os

data_path = '/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_int_1frame/'
save_path = '/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_int_1frame_numpy/'

# Find all files that match the pattern
file_pattern = data_path + 'Date*_Sub*_*.pkl'
all_files = glob.glob(file_pattern)

# Filter out files containing 'box'
filtered_files = [f for f in all_files if 'box' not in os.path.basename(f)]

for file_path in filtered_files:

    print(file_path)
    # Process each file
    with open(file_path, 'rb') as f:
        while True:
            try:
                objects = pickle.load(f)
            except EOFError:
                break

    for j in range(len(objects)):
        for k in range(len(objects[j])):
            a = objects[j][k]
            for key in a.keys():
                if hasattr(a[key], 'numpy'):
                    objects[j][k][key] = a[key].numpy()
                else:
                    objects[j][k][key] = a[key]

    # Extracting the specific file name from the path
    file_name = file_path.split('/')[-1]
    with open(save_path + file_name, 'wb') as f:
        # Write the modified object to a new pickle file
        pickle.dump(objects, f)


