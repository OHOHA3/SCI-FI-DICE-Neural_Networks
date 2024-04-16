import albumentations as A
from PIL import Image


def resolution(index, file):
    with Image.open(file) as pic:
        print(index, "-", pic.size[0], "x", pic.size[1])


def transform1():
    transform = A.Compose([
        A.HorizontalFlip(p=1),
    ])
    return transform


def transform2():
    transform = A.Compose([
        A.Affine(rotate=15, p=1),
    ])
    return transform


def transform3():
    transform = A.Compose([
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=10, num_shadows_upper=15, p=1)
    ])
    return transform


def transform4():
    transform = A.Compose([
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.3, p=1),
    ])
    return transform


def transform5():
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), contrast_limit=(0.3, 0.3), brightness_by_max=True, p=1),
    ])
    return transform


def transform6():
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_by_max=True, p=1),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=10, num_shadows_upper=15, p=1),
        A.Affine(rotate=15, p=1),
        A.HorizontalFlip(p=1),
    ])
    return transform
