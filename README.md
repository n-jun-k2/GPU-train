# Nvidia-Docker-train
Sample to develop a program using GPU on WSL2 Ubuntu.

## Setup WSL2
Minimize the WSL2 Ubuntu environment([wsl_setup.md](./Doc/wsl_setup.md))

## WSL operation
```bash
# Login to WSL
> wsl
# Root login
> sudo -i
[sudo] password for root:
# start docker
> service docker start
```

## How to run
Write the image and execution of each container to the makefile as needed.

| service name | overview | volume path |
|---|---|---|
| python | ubuntu22.04+pytorch(2.1.0+cu118) environment| ./Docker/python/packages/ |
| cpp | ubuntu22.04+cudnn8+nvcc+boost+PyTorch environment | ./Docker/cpp/src/ |
| q | Q# Execution environment + Python environment | ./Docker/q/packages/ , ./Dockerq/src/|

```bash
# launch a container 
user:GPU-train# docker compose up -d <service name>

## Example python container
user:GPU-train# docker compose up -d python

[+] Running 2/2
 ✔ Network gpu-train_default     Created                                                                                               0.0s 
 ✔ Container gpu-train-python-1  Started                                                                                               0.0s
user:GPU-train$ docker compose exec python bash
root@b1dfb96230f6:/packages#
```

## Folder structure
```
GPU-TRAIN
    ┣ Doc/・・・・・・・・・・・・ Document
    ┃   ┗ ...
    ┣ Docker/
    ┃   ┣ c /・・・・・・・・・・・ Development of CUDA
    ┃   ┃   ┣ src /
    ┃   ┃   ┃   ┣ ...
    ┃   ┃   ┃   ┗ ...
    ┃   ┃   ┣ Dockerfile
    ┃   ┃   ┗ README.md
    ┃   ┣  python /・・・・・・・・ Project folder developed with python
    ┃   ┃    ┣ packages / Contains the application.
    ┃   ┃    ┃   ┣ ...
    ┃   ┃    ┃   ┗ ...
    ┃   ┃    ┣ .gitignore
    ┃   ┃    ┣ Dockerfile
    ┃   ┃    ┗ requirements.txt
    ┃   ┗  q /
    ┃      ┣ packages /
    ┃      ┣ src /
    ┃      ┣ dockerfile
    ┃      ┗ requirements.txt
    ┣ scripts
    ┣ LICENSE
    ┣ makefile
    ┗ README.md
```