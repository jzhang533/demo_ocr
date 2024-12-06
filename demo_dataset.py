import os
from pprint import pprint
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, Optional
from torchvision import transforms
from transformers import (AutoProcessor, AutoTokenizer, default_data_collator)


IMAGES_DIR='./data/images'
CAPTION_FILE='./data/labels.txt'

class MyDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        processor = None,
        tokenizer = None,
    ):
        """
        Args:
            image_dir (str): Directory containing the images
            caption_file (str): Path to text file containing captions
        """
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        # Read captions and image filenames
        self.items = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming each line is in format: image_filename.jpg|caption
                # Modify the split character based on your file format
                image_name, caption = line.strip().split('|')
                image_path = os.path.join(image_dir, image_name)
                
                # Only add if image file exists
                if os.path.exists(image_path):
                    self.items.append({
                        'image_path': image_path,
                        'caption': caption
                    })
                else:
                    raise Exception("image {} not exists".format(image_path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        
        # Load and convert image to RGB
        image = Image.open(item['image_path']).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        text = item['caption']
        labels = self.tokenizer(
            text,
            padding="max_length",
            max_length=32,
            truncation=True,
        ).input_ids
        labels = np.array(labels)
        # important: make sure that PAD tokens are ignored by the loss function
        #labels[labels == self.tokenizer.pad_token_id] = -100

        encoding = {
            "pixel_values": pixel_values,
            "labels":torch.tensor(labels)
        }

        return encoding

# Example usage:
if __name__ == "__main__":
    
    processor = AutoProcessor.from_pretrained('facebook/deit-tiny-patch16-224', use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Create dataset
    dataset = MyDataset(
        image_dir=IMAGES_DIR,
        caption_file=CAPTION_FILE,
        processor=processor,
        tokenizer=tokenizer
    )

    for i in range(4):
        sample = dataset[i]
        print(sample['labels'].shape)
        print(sample['pixel_values'].shape)    

    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True,
        collate_fn=default_data_collator
    )

    for batch_idx, batch in enumerate(dataloader):
        print('=' * 80)
        print(batch_idx)
        print(batch['labels'].shape)
        print(batch['pixel_values'].shape)
        #pprint(batch)
        print('=' * 80)



