import subprocess

# upgrade pip, if necessary
subprocess.run(["sudo", "pip", "install", "--upgrade", "pip", "protobuf==3.19.6"])

# configure git credential helper
subprocess.run(["git", "config", "--global", "credential.helper", "store"])

# install 🤗 libraries
subprocess.run(
    [
        "sudo",
        "pip",
        "install",
        "--upgrade",
        "git+https://github.com/huggingface/datasets",
        "git+https://github.com/huggingface/transformers",
        "evaluate",
        "huggingface_hub",
        "jiwer",
        "bitsandbytes",
        "accelerate",
    ]
)

# uninstall torchvision cu11.7
subprocess.run(["sudo", "pip", "uninstall", "torchvision"])

# install torchvisino cu11.3 and diffusers
subprocess.run(["sudo", "pip", "install", "torchvision", "diffusers"])
