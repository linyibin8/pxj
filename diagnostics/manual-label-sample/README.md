# Manual Label Sample

This folder is an isolated sample annotation set created for review. It does not modify the existing YOLO train/val/test dataset.

Class mapping:

- `0`: `question_block`

Labeling convention used in this sample:

- One box covers one complete question block or a continuous sub-question block.
- Include the printed prompt, diagrams, and visible student answer work that belongs to that block.
- Exclude large unrelated background areas such as desk, keyboard, margins, and empty paper whenever practical.
- Negative examples use an empty `.txt` label file.

Folders:

- `images/`: copied source images
- `labels/`: YOLO-format labels
- `previews/`: rendered images with red boxes

