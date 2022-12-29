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
