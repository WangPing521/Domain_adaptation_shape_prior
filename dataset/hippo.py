from pathlib import Path

from augment.synchronize import SequentialWrapper
from ._ioutils import downloading
from .base import DatasetBase



class HippocampusDataset(DatasetBase):
    download_link = "https://drive.google.com/uc?id=1VG14fqf6EltsR7HUs5dFvN0X7ru0w_wH"  # //spleen link
    zip_name = "Hippocampus.zip"
    folder_name = "Hippocampus"

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        sub_folders = ["img", "gt"]
        sub_folder_types = ["image", "gt"]
        group_re = r"hippocampus_\d+"
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)

        super().__init__(root_dir=str(path), mode=mode, sub_folders=sub_folders, sub_folder_types=sub_folder_types,
                         transforms=transforms, group_re=group_re)
