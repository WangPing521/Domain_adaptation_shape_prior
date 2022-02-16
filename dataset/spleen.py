from pathlib import Path

from augment.synchronize import SequentialWrapper
from ._ioutils import downloading
from .base import DatasetBase


class SpleenDataset(DatasetBase):
    download_link = "https://drive.google.com/file/d/1BkZcYU1Dnp1soVz9tTQedoks3gxiOn6-/view?usp=sharing"
    zip_name = "Spleen.zip"
    folder_name = "Spleen"

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        sub_folders = ["img", "gt"]
        sub_folder_types = ["image", "gt"]
        group_re = r"spleen_\d+"
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)

        super().__init__(root_dir=str(path), mode=mode, sub_folders=sub_folders, sub_folder_types=sub_folder_types,
                         transforms=transforms, group_re=group_re)
