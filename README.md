# Nvidia-Docker-train
Sample to develop a program using GPU on WSL2 Ubuntu.

## Setup WSL2
Minimize the WSL2 Ubuntu environment([wsl_setup.md](./Doc/wsl_setup.md))


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
    ┣ Doc/・・・・・・・・・・・・ ドキュメント成果物
    ┃   ┗
    ┣ Docker/
    ┃   ┣ c /・・・・・・・・・・・ C言語で開発するプロジェクトフォルダ
    ┃   ┃   ┣ src /
    ┃   ┃   ┃   ┣ ...
    ┃   ┃   ┃   ┗ ...
    ┃   ┃   ┣ Dockerfile
    ┃   ┃   ┗ README.md
    ┃   ┗ python /・・・・・・・・ pythonで開発するプロジェクトフォルダ
    ┃       ┣ packages / アプリケーションが格納されている。
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