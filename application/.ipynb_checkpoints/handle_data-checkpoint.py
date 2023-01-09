import glob
import os

import transformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothData(Dataset):
    def __init__(self, folder_path: str, tokenizer, instance_prompt: str, size=512):
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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
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
