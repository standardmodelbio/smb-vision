#!/bin/bash

# First, create a script to extract paths from JSON
python3 - << EOF
import json

with open('./smb-vision-large-train-mim.json') as f:
    data = json.load(f)

# Extract paths and save them to a file
with open('s3_paths.txt', 'w') as out:
    for sample in data['train'] + data['validation']:
        path = sample['image']
        s3_path = "datasets/idc2niix/" + "/".join(path.split("/")[2:])
        out.write(f"{s3_path}\n")
EOF

# Create a directory for the files
mkdir -p ../nifti_files

# Read each path and sync individually
while IFS= read -r path; do
    aws s3 cp s3://smb-dev-us-east-2-data/$path ../nifti_files/$path
done < s3_paths.txt

# Clean up temporary file
rm s3_paths.txt
