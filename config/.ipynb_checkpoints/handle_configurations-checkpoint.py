import argparse


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Configuration settings for training run.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="pretrained_model",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="Training device (e.g. 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3434554,
        required=False,
        help="Random seed value for stability across training and inference runs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        required=False,
        help="Number of steps where gradients are accumulated over multiple mini-batches.",
    )
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=3e-06,
        required=False,
        help="The step size at which the optimizer makes updates to the model parameters during training.",
    )
    parser.add_argument(
        "--train_data_directory",
        type=str,
        default="input_train_images",
        required=False,
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        required=False,
        help="Number of training examples that are processed in a single iteration by the model.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=209,
        required=False,
        help="The number of times the model processes a batch of training data during the training process.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=int,
        default=1.0,
        required=False,
        help="The maximum allowable size of the gradient vector during training.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="output_model",
        required=False,
        help="Path to directory where model weights are to be saved after training.",
    )
    parser.add_argument(
        "--image_concept",
        type=str,
        default=None,
        required=True,
        help="Image concept. Typically features animal type followed by 'pokemon' (e.g. german shorthaired pokemon).",
    )
    parser.add_argument(
        "--image_object",
        type=str,
        default=None,
        required=True,
        help="Image object. The type of object being generated (e.g. 'dog', 'cat', etc.)",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default="n",
        choices=["y", "n"],
        required=False,
        help="Customized prompt appended to image concept and object type (e.g. <german shorthaired pokemon> <dog> wearing white gloves).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        required=False,
        help="Number of image denoising steps. 50-200 returns the best results, experimentally.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=10,
        required=False,
        help="Paramater that controls how much the image generation process follows the text prompt. Default is 10.",
    )
    parser.add_argument(
        "--experiment_results",
        type=str,
        default="output_images",
        required=False,
        help="Path to directory where generated images are to be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        required=False,
        help="Resolution of generated images. Default is 512.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        required=False,
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--num_images_to_generate",
        type=int,
        default=5,
        required=False,
        help="Number of images to generate per inference call."
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default="tensorboard",
        required=False,
        choices=['all', 'aim', 'tensorboard', 'wandb', 'comet_ml', 'mlflow'],
        help="Supported logging capabilities.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        required=False,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--num_of_gpus",
        type=int,
        default=1,
        required=False,
        help=("Number of gpus to use for training."),
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args