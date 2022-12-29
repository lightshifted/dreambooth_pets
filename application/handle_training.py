from argparse import Namespace
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import notebook_launcher
import math
from tqdm.auto import tqdm
from pathlib import Path
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from application.handle_data import DreamBoothData
from config.handle_configurations import parse_args

logger = get_logger(__name__)


class Trainer:
    def __init__(self, args: Namespace, custom_prompt: bool=False):
                
        self.args = args
        self.log_dir = Path(self.args.output_directory, self.args.logging_dir)
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.args.model_id, torch_dtype=torch.float32)
        self.pipeline.to(self.args.device)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            log_with=self.args.log_with,
            logging_dir='logs',
        )
        
        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.feature_extractor = self.pipeline.feature_extractor
        
        self.optimizer_class = torch.optim.AdamW
        self.optimizer = self.optimizer_class(
            self.unet.parameters(),  # only optimize unet
            lr=self.args.learning_rate,
        )
        self.noise_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            num_train_timesteps=1000,
        )
        
        self.instance_prompt = f"{self.args.image_concept} {self.args.image_object}"
            
        self.train_dataset = DreamBoothData(
            self.args.train_data_directory,
            self.tokenizer,
            self.instance_prompt,
            size=self.args.resolution,
        )        
            
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
                    
        self.unet, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader
        )
        
        # Move text_encode and vae to gpu
        self.text_encoder.to(self.accelerator.device)
        self.vae.type(torch.FloatTensor)
        self.vae.to(self.accelerator.device)
                
        # Set the seed for the random number generator, if specified
        if self.args.seed is not None:
            set_seed(self.args.seed)
            
        def training_function():
            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(
                len(self.train_dataloader) / self.args.gradient_accumulation_steps
            )
            num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

            # Train!
            total_batch_size = (
                self.args.train_batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
            )
            
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(self.train_dataset)}")
            logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
            logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation). = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {args.max_train_steps}")
            logger.info(f"  Num of inference steps = {args.num_inference_steps}")
            logger.info(f"  Guidance scale: ")
            logger.info(f"  Learning rate: ")
            logger.info(f"  Inference steps: ")

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(
                range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process
            )
            progress_bar.set_description("Steps")
            global_step = 0
            for epoch in range(num_train_epochs):
                self.unet.train()
                for step, batch in enumerate(self.train_dataloader):
                    with self.accelerator.accumulate(self.unet):
                        # Convert images to latent space
                        with torch.no_grad():
                            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
                            latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn(latents.shape).to(latents.device)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        ).long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        with torch.no_grad():
                            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                        # Predict the noise residual
                        noise_pred = self.unet(
                            noisy_latents, timesteps, encoder_hidden_states
                        ).sample
                        loss = (
                            F.mse_loss(noise_pred, noise, reduction="none")
                            .mean([1, 2, 3])
                            .mean()
                        )
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                    logs = {"loss": loss.detach().item()}
                    progress_bar.set_postfix(**logs)

                    if global_step >= self.args.max_train_steps:
                        break

                self.accelerator.wait_for_everyone()

            # Create the pipeline using using the trained modules and save it.
            if self.accelerator.is_main_process:
                print(f"Loading pipeline and saving to {self.args.output_directory}...")
                scheduler = PNDMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    skip_prk_steps=True,
                    steps_offset=1,
                )
                pipeline = StableDiffusionPipeline(
                    text_encoder=self.text_encoder,
                    vae=self.vae,
                    unet=self.accelerator.unwrap_model(self.unet),
                    tokenizer=self.tokenizer,
                    scheduler=scheduler,
                    safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                        "safety_checker"
                    ),
                    feature_extractor=self.feature_extractor,
                )
                pipeline.save_pretrained(self.args.output_directory)
        
        notebook_launcher(
            training_function, num_processes=self.args.num_of_gpus # CHANGE THIS TO MATCH THE NUMBER OF GPUS YOU HAVE
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            self.args.output_directory,
            torch_dtype=torch.float32,
        ).to("cuda")

        if self.args.custom_prompt == "y":
            prompt = input("Input text prompt: ")
            prompt = f"{self.args.image_concept} {self.args.image_object} {prompt}"
        else:
            prompt = self.instance_prompt

        all_images = []
        for _ in range(self.args.num_images_to_generate):
            images = pipe(prompt, guidance_scale=self.args.guidance_scale, num_inference_steps=self.args.num_inference_steps).images
            all_images.extend(images)

        title = str(self.args.learning_rate) + "_" + str(self.args.max_train_steps) + "_" + str(self.args.guidance_scale)
        for idx, im in enumerate(all_images):
            # if not os.path.exists(f"{self.args.output_directory}/{idx:03}_{title}_{args.image_object}"):
            #     os.makedirs(f"{self.args.output_directory}/{idx:03}_{title}_{args.image_object}")
            im.save(f"{self.args.experiment_results}/{idx:03}_{title}_{self.args.image_object}.png")
        
        # free up memory
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
                
    def collate_fn(self, examples):
        # Extract the "instance_prompt_ids" and "instance_images" fields from each example
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Stack the "instance_images" and convert them to a tensor of floating point values
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # Pad the input IDs and convert them to a tensor
        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids

        # Create a dictionary with the input tensors
        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch
    
    
if __name__ == "__main__":
    args = parse_args()
    Trainer(args)