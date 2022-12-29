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
Step 4: Download and install the latest release of Cog directly from GitHub by running the following commands in a terminal:

```bash
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```
Step 5: Initialize cog:
```bash
sudo cog init
```
Step 6: Run each cell in the `get_weights.ipynb` notebook to collect pretrained weights from HuggingFace repositories and to store them in a local directory.