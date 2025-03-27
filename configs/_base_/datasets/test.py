import os

train_path = '../../../data/imagenet/train'
val_path = '../../../data/imagenet/val'

print("Train folder exists:", os.path.exists(train_path))
print("Validation folder exists:", os.path.exists(val_path))

# 检查文件夹是否有内容
print("Train folder files:", os.listdir(train_path)[:5] if os.path.exists(train_path) else "Not found")
print("Validation folder files:", os.listdir(val_path)[:5] if os.path.exists(val_path) else "Not found")
