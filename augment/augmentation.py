from augment import pil_augment
from augment.synchronize import SequentialWrapper

train_transforms = SequentialWrapper(
    img_transform=pil_augment.Compose(
        [
            pil_augment.Resize((256, 256)),
            pil_augment.RandomCrop((224, 224)),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomRotation(degrees=10),
            pil_augment.ToTensor(),
        ]
    ),
    target_transform=pil_augment.Compose(
        [
            pil_augment.Resize((256, 256)),
            pil_augment.RandomCrop((224, 224)),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomRotation(degrees=10),
            pil_augment.ToLabel(),
        ]
    ),
    if_is_target=(False, True),
)
val_transform = SequentialWrapper(
    img_transform=pil_augment.Compose(
        [pil_augment.CenterCrop((224, 224)), pil_augment.ToTensor()]
    ),
    target_transform=pil_augment.Compose(
        [pil_augment.CenterCrop((224, 224)), pil_augment.ToLabel()]
    ),
    if_is_target=(False, True),
)