import subprocess

# Prompt the user to enter the name of their virtual environment
env_name = input("Enter the name of your virtual environment: ")

# Set up a list of commands to be run                 
commands = [
    "sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4",
    "sudo apt update",
    "sudo apt install -y ffmpeg",
    "sudo apt-get install git-lfs",
    f"python3 -m venv {env_name}",
    f"echo \"source ~/{env_name}/bin/activate\" >> ~/.bashrc",
    "bash"
]

# Iterate through the list of commands and run each one
for command in commands:
    subprocess.run(command, shell=True)