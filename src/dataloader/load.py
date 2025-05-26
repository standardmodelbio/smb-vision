import json
import shutil
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import monai
import pandas as pd
import torch
from monai.data.utils import SUPPORTED_PICKLE_MOD
from monai.utils import look_up_option
from PIL import Image
from torch.utils.data import Dataset

from transformers import AutoProcessor

from .transforms import ct_transforms


def load_data(file_path: Union[str, Path], split: str = None) -> List[Dict]:
    """Load data from various file formats (json, csv, parquet).

    Args:
        file_path (Union[str, Path]): Path to the data file
        split (str, optional): If data is split into train/val/test, specify which split to load

    Returns:
        List[Dict]: List of data items as dictionaries
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle different file formats
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
            if split and isinstance(data, dict):
                if split not in data:
                    raise ValueError(f"Split '{split}' not found in data. Available splits: {list(data.keys())}")
                return data[split]
            return data if isinstance(data, list) else list(data.values())

    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
        if split and "split" in df.columns:
            df = df[df["split"] == split]
        return df.to_dict("records")

    elif file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
        if split and "split" in df.columns:
            df = df[df["split"] == split]
        return df.to_dict("records")

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .json, .csv, .parquet")


class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform, cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
        print(f"Size of dataset: {self.__len__()}\n")

    def _cachecheck(self, item_transformed):
        """
        A function to cache the expensive input data transform operations
        so that huge data sets (larger than computer memory) can be processed
        on the fly as needed, and intermediate results written to disk for
        future use.

        Args:
            item_transformed: The current data element to be mutated into transformed representation

        Returns:
            The transformed data_element, either from cache, or explicitly computing it.

        Warning:
            The current implementation does not encode transform information as part of the
            hashing mechanism used for generating cache names when `hash_transform` is None.
            If the transforms applied are changed in any way, the objects in the cache dir will be invalid.

        """
        hashfile = None
        if self.cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            data_item_md5 += self.transform_hash
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            try:
                return torch.load(hashfile, weights_only=False)
            except PermissionError as e:
                if sys.platform != "win32":
                    raise e
            except RuntimeError as e:
                if "Invalid magic number; corrupt file" in str(e):
                    warnings.warn(f"Corrupt cache file detected: {hashfile}. Deleting and recomputing.")
                    hashfile.unlink()
                else:
                    raise e

        _item_transformed = self._pre_transform(deepcopy(item_transformed))  # keep the original hashed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_item_transformed,
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


class SMBVisionDataset(CTPersistentDataset):
    def __init__(
        self, data_path: Union[str, Path], split: str = "train", transform=ct_transforms["smb-vision"], cache_dir=None
    ):
        """Initialize SMBVision dataset.

        Args:
            data_path (Union[str, Path]): Path to the data file (json, csv, or parquet)
            split (str, optional): Data split to use. Defaults to "train"
            transform: Transform to apply to the data
            cache_dir: Directory for caching transformed data
        """
        data = load_data(data_path, split=split)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class MIMDataset(monai.data.PersistentDataset):
    def __init__(self, data_path: Union[str, Path], transform=ct_transforms["mim"], cache_dir=None):
        """Initialize MIM dataset.

        Args:
            data_path (Union[str, Path]): Path to the data file (json, csv, or parquet)
            transform: Transform to apply to the data
            cache_dir: Directory for caching transformed data
        """
        data = load_data(data_path)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
        print(f"Size of dataset: {self.__len__()}\n")


class SiglipDataset(Dataset):
    """Dataset class for loading and preprocessing x-ray images for SigLIP model"""

    def __init__(self, model_id: str, data_path: Union[str, Path], cache_dir: str = None):
        """Initialize the dataset.

        Args:
            model_id (str): The model ID to use for the processor
            data_path (Union[str, Path]): Path to the data file (json, csv, or parquet)
            cache_dir (str, optional): Directory for caching processed images
        """
        self.data_dict = self._validate_data(load_data(data_path))
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
