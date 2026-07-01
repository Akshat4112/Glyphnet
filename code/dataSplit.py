import shutil
import os
import argparse

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, default="../data", help="Define path for the data")
BASE_PATH = parser.parse_args().path_data


def split_class(class_name):
    """Move rendered images for one class into train/valid/test at 70/20/10.

    Files are sorted first so the split is reproducible across runs (os.listdir
    order is otherwise undefined).
    """
    src = os.path.join(BASE_PATH, class_name)
    files = sorted(os.listdir(src))
    total = len(files)
    print(class_name, total)

    n_train = int(0.7 * total)
    n_valid = int(0.2 * total)
    # test gets the remainder so every file lands in exactly one split
    splits = {
        'train': files[:n_train],
        'valid': files[n_train:n_train + n_valid],
        'test': files[n_train + n_valid:],
    }

    for split, split_files in splits.items():
        dst = os.path.join(BASE_PATH, split, class_name)
        os.makedirs(dst, exist_ok=True)
        for item in split_files:
            shutil.move(os.path.join(src, item), dst)

    # only remove the source dir if the split consumed everything
    remaining = os.listdir(src)
    if not remaining:
        os.rmdir(src)
    else:
        print("Not removing %s: %d files remain" % (src, len(remaining)))


for name in ('real', 'fake'):
    split_class(name)
