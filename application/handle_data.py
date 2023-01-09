import glob
import os
from typing import Dict, List

import torch
import torchvision.transforms as transforms
import transformers
from PIL import Image
from torch.utils.data import Dataset


class DreamBoothData(Dataset):
    """
    A PyTorch Dataset class for loading and processing
    images and text data for the DreamBooth project.

    Parameters
    ----------
    folder_path: str
        The path to the directory containing the images.
    tokenizer: transformers.PreTrainedTokenizer
        The tokenizer to use for encoding the text data.
    instance_prompt: str
        The prompt to use for each instance.
    size: int, optional (default=512)
        The size of the images in the dataset.
    """

    def __init__(
        self,
        folder_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        instance_prompt: str,
        size: int = 512,
    ):
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "lambdalabs/sd-pokemon-diffusers", subfolder="tokenizer"
        )

        self.transformer = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.file_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(
            os.path.join(folder_path, "*.png")
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset, which is the number of images in the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns the example at the given index.

        Parameters
        ----------
        index: int
            The index of the example to return.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the image data and the encoded text data.
        """
        images = []
        # Iterate over the list of file_paths and load the images
        for file_path in self.file_paths:
            with Image.open(file_path) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Convert the image to a tensor using the transformer object
                image = self.transformer(image)

                # Add the image to the list
                images.append(image)

        example = {}
        example["instance_images"] = images[index]
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example
