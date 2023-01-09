
## CLI
Using the CLI, we can define the model, training and data arguments for fine-tuning.
```bash
--model-id="pretrained_model",
--device="cuda",
--seed=3434553,
--gradient_accumulation_steps=1,
--learning_rate=0.000003,
--train_data_directory="input_train_images",
--train_batch_size=2,
--max_train_steps=209,
--max_grad_norm=1.0,
--output_directory="output_model",
--image_concept="shih tzu",
```