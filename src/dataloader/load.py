import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import monai
import torch
from merlin.data.monai_transforms import ImageTransforms
from monai.data.utils import SUPPORTED_PICKLE_MOD
from monai.utils import look_up_option
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from .transforms import ct_transforms


class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform, cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")

    def _cachecheck(self, item_transformed):
        hashfile = None
        _item_transformed = deepcopy(item_transformed)
        image_data = {"image": item_transformed.get("image")}  # Assuming the image data is under the 'image' key

        if self.cache_dir is not None and image_data is not None:
            data_item_md5 = self.hash_func(image_data).decode("utf-8")  # Hash based on image data
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():
            cached_image = torch.load(hashfile)
            _item_transformed["image"] = cached_image
            return _item_transformed

        _image_transformed = self._pre_transform(image_data)["image"]
        _item_transformed["image"] = _image_transformed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_image_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)


class MerlinDataset(CTPersistentDataset):
    def __init__(self, data, transform=ct_transforms["merlin"], cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")


class SMBVisionDataset(CTPersistentDataset):
    def __init__(self, data, transform=ct_transforms["smb-vision"], cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")


class MIMDataset(CTPersistentDataset):
    def __init__(self, data, transform=ct_transforms["mim"], cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")


class DataLoader(monai.data.DataLoader):
    def __init__(
        self,
        datalist: List[dict],
        cache_dir: str,
        batchsize: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.datalist = datalist
        self.cache_dir = cache_dir
        self.batchsize = batchsize
        self.dataset = CTPersistentDataset(
            data=datalist,
            transform=ImageTransforms,
            cache_dir=cache_dir,
        )
        super().__init__(
            self.dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
        )


class SiglipDataset(Dataset):
    """Dataset class for loading and preprocessing x-ray images for SigLIP model"""

    def __init__(self, model_id: str, data_dict: List[Dict], cache_dir: str = None):
        """Initialize the dataset.

        Args:
            model_id (str): The model ID to use for the processor
            data_dict (List[Dict]): List of dictionaries containing image data
            cache_dir (str, optional): Directory for caching processed images
        """
        self.data_dict = self._validate_data(data_dict)
        # self.data_dict = data_dict
        self.cache_dir = cache_dir
        self.processor = AutoProcessor.from_pretrained(f"google/{model_id}")

    def _validate_data(self, data_dict: List[Dict]) -> List[Dict]:
        """Validate the input data dictionary using multithreading.

        Args:
            data_dict (List[Dict]): List of dictionaries containing image data

        Returns:
            List[Dict]: Validated data dictionary
        """
        if not isinstance(data_dict, list):
            raise ValueError("Input data must be a list of dictionaries")

        def validate_item(item: Dict) -> Dict:
            """Validate a single data item.

            Args:
                item (Dict): Dictionary containing image data

            Returns:
                Dict: Validated item or None if invalid
            """
            if not isinstance(item, dict):
                return None

            if "uid" not in item or "image_path" not in item:
                return None

            image_path = Path(item["image_path"])
            if not image_path.exists():
                return None

            try:
                with Image.open(image_path) as img:
                    img.verify()
                return item
            except Exception as e:
                print(f"Warning: Invalid image file {image_path}: {str(e)}")
                return None

        # Use ThreadPoolExecutor for parallel validation
        validated_data = []
        with ThreadPoolExecutor(max_workers=min(32, len(data_dict))) as executor:
            # Submit all validation tasks
            future_to_item = {executor.submit(validate_item, item): item for item in data_dict}

            # Process results as they complete
            for future in as_completed(future_to_item):
                result = future.result()
                if result is not None:
                    validated_data.append(result)

        if not validated_data:
            raise ValueError("No valid images found in the input data")

        return validated_data

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.data_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item to get

        Returns:
            Dict[str, Any]: Dictionary containing the processed image and metadata
        """
        data = self.data_dict[idx]
        uid = data["uid"]
        image_path = data["image_path"]

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        image = inputs.pixel_values.squeeze(0)

        return {
            "uid": uid,
            "image": image,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for the DataLoader.

        Args:
            batch (List[Dict[str, Any]]): List of items from the dataset

        Returns:
            Dict[str, Any]: Batched data
        """
        uids = [item["uid"] for item in batch]
        images = torch.stack([item["image"] for item in batch])

        return {
            "uid": uids,
            "image": images,
        }
