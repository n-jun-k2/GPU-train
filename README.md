# Nvidia-Docker-train
このプロジェクトでは、WSL2上のUbuntuを利用したDockerのGPU処理を活用するサンプル。

## WSL2のUbuntuのセットアップ
[wsl_setup.md](./Doc/wsl_setup.md)


## コンテナ起動方法
run時に`--gpus`でgpuを指定する。
```
docker run --gpus all --name <container-name> -it <image-name> /bin/bash
```