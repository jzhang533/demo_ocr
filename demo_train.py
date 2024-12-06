import os
from pprint import pprint
from typing import Dict, Optional
from PIL import Image

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision import transforms
from transformers import (AutoProcessor, AutoTokenizer, BertConfig, GPT2Config,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          VisionEncoderDecoderConfig,
                          VisionEncoderDecoderModel, ViTConfig,
                          default_data_collator)

config_encoder = ViTConfig()
config_decoder = GPT2Config()

config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)

processor = AutoProcessor.from_pretrained('facebook/deit-tiny-patch16-224', use_fast=True)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

from demo_dataset import CAPTION_FILE, IMAGES_DIR, MyDataset

# Create dataset
dataset = MyDataset(
    image_dir=IMAGES_DIR,
    caption_file=CAPTION_FILE,
    processor=processor,
    tokenizer=tokenizer,
)

run_name = "debug"
batch_size = 2
num_epochs = 100
max_len = 32
fp16 = True


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="no",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=fp16,
    fp16_full_eval=fp16,
    dataloader_num_workers=16,
    output_dir="./outputs",
    logging_steps=10,
    report_to="none",
    save_steps=200,
    #eval_steps=200,
    num_train_epochs=num_epochs,
    run_name=run_name,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    #processing_class = 
    args=training_args,
    train_dataset=dataset,
    #eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()

# Load and convert image to RGB
image = Image.open("{}/00.jpg".format(IMAGES_DIR)).convert('RGB')

inputs = processor(images=image, return_tensors="pt")
print("shape of inputs")
pprint(inputs.pixel_values.shape)

generated_ids = model.cpu().generate(inputs.pixel_values, bos_token_id=tokenizer.bos_token_id, max_new_tokens=20)[0].cpu()

print(generated_ids)
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(output_text)