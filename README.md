## Set Up an Environment
Estimated time to complete: 5 mins
In this section, we'll cover how to set up an environment with the required libraries. This section assumes that you are SSH'd into your GPU device.

Step 1: Upload lambda_dev_env.zip to Lambda Cloud's Jupyter Lab instance and unzip its contents into the main directory.
```bash
$ unzip lamda_dev_env.zip
```

Step 2: Set up virtual environment:
```bash
$ python install_db_env.py
```

Step 3: Install libraries
```bash
$ sudo python -m pip install -e .
```

Step 4: Run each cell in the `get_weights.ipynb` notebook to collect pretrained weights from HuggingFace repositories and to store them in a local directory.

# This project offers great flexiblity when it comes to running the model. Here are the options on offer:

## Replicate (Cog)


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

## Replicate (Cog)
Step 5: Download and install the latest release of Cog directly from GitHub by running the following commands in a terminal
```bash
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```
Step 6: Initialize cog
```bash
sudo cog init
```
Step 7: Run the model
```bash
cog run predict.py --model-id="pretrained_model" --device="cuda" --seed=3434553 --gradient_accumulation_steps=1 --learning_rate=0.000003 --train_data_directory="input_train_images" --train_batch_size=2 --max_train_steps=209 --max_grad_norm=1.0 --output_directory="output_model" --image_concept="shih tzu"
```

## API
Step 5: Run model using FastAPI
```bash
uvicorn app.api:app --host 0.0.0.0 --port=8000 --reload --reload-dir app
```
Step 6: Generate predictions
```bash
curl -X GET http://localhost:8000/predict
```
