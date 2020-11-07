# デバイスの管理
Nvidiaでは、GPUデバイスの情報の取得や管理を行う手法がいくつかあります。
実行時にカーネルの実行設定を定義するのに役立つため、その取得方法を理解する。

- CUDA Runtime APIの関数
- Nvidia Systems Management Interface (nvidia-smi) コマンドライン

今回のサンプルではCUDA Runtime APIから取得する。
