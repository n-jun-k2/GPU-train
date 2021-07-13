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

### for python 
```bash
# Build Docker image
make build name=python
# Start and login container
make python-run
```
### for C lang
```bash
# Build Docker image
make build name=c
# Start and login container
make c-run
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
    ┃   ┗ python /・・・・・・・・ Project folder developed with python
    ┃       ┣ packages / Contains the application.
    ┃       ┃   ┣ ...
    ┃       ┃   ┗ ...
    ┃       ┣ .gitignore
    ┃       ┣ Dockerfile
    ┃       ┗ requirements.txt
    ┃
    ┣ scripts
    ┣ LICENSE
    ┣ makefile
    ┗ README.md
```