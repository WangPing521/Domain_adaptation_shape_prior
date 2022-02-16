from torch import Tensor

from utils.utils import fix_all_seed_for_transforms


class RisingWrapper:

    def __init__(
            self,
            *,
            geometry_transform,
            intensity_transform
    ) -> None:
        self.geometry_transform = geometry_transform
        self.intensity_transform = intensity_transform

    def __call__(self, image: Tensor, *, mode: str, seed: int):
        assert mode in ("image", "feature"), f"`mode` must be in `image` or `feature`, given {mode}."
        if mode == "image":
            with fix_all_seed_for_transforms(seed):
                if self.intensity_transform is not None:
                    image = self.intensity_transform(data=image)["data"]
            with fix_all_seed_for_transforms(seed):
                if self.geometry_transform is not None:
                    image = self.geometry_transform(data=image)["data"]
        else:
            with fix_all_seed_for_transforms(seed):
                if self.geometry_transform is not None:
                    image = self.geometry_transform(data=image)["data"]
        return image
