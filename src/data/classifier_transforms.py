"""Image-only transforms for the AcneSCU crop classifier. Crops are saved
to disk at their original (padded/clamped) resolution — resizing to the
model's input size happens here, at load time, not permanently."""
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_classifier_transform(train: bool, input_size: int = 224) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
