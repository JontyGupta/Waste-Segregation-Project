import shutil, random
from pathlib import Path

random.seed(42)
img_train = Path('data/images/train')
lbl_train = Path('data/labels/train')
img_val   = Path('data/images/val')
lbl_val   = Path('data/labels/val')

img_val.mkdir(parents=True, exist_ok=True)
lbl_val.mkdir(parents=True, exist_ok=True)

images = list(img_train.glob('*.*'))
random.shuffle(images)
val_images = images[:int(len(images) * 0.2)]

for img in val_images:
    shutil.move(str(img), img_val / img.name)
    lbl = lbl_train / (img.stem + '.txt')
    if lbl.exists():
        shutil.move(str(lbl), lbl_val / lbl.name)

print(f'Moved {len(val_images)} images to val')
print(f'Train remaining: {len(list(img_train.glob("*.*")))}')
print(f'Val images: {len(list(img_val.glob("*.*")))}')