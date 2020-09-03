# CUDAのテスト

## CUDAの実行
```
nvcc -arch <最適化アーキテクチャ> <ソースファイル> -o <出力ファイル名>
```

今回の環境ではRTX2060の環境のため以下のように実行する。
```bash
> nvcc main.cu -o hello -arch=sm_75
> ./hello 
Hello World from CPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
```

## プロファイリング
``` 
> nvprof ./hello
Hello World from CPU!
==64== NVPROF is profiling process 64, command: ./hello
==64== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory       
==64== Error: Internal profiling error 4075:9.
======== Profiling result:
No kernels were profiled.
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
      API calls:   99.88%  170.07ms         1  170.07ms  170.07ms  170.07ms  cudaLaunchKernel
                    0.11%  183.00us         1  183.00us  183.00us  183.00us  cuDeviceTotalMem
                    0.01%  13.400us        97     138ns     100ns     600ns  cuDeviceGetAttribute
                    0.00%  3.5000us         2  1.7500us     200ns  3.3000us  cuDeviceGet
                    0.00%  1.6000us         3     533ns     100ns  1.3000us  cuDeviceGetCount
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
======== Error: CUDA profiling error.
```

## コマンドオプション
[CUDA doc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options)

| オプション | 概要 |
|:---:|-|
| -o | 出力ファイルの名前と場所を指定します。 |
| -include | 事前に含める必要があるヘッダーファイルを指定します。 |
| -l | リンクステージで使用するライブラリを、ライブラリファイル拡張子なしで指定します。ライブラリは、オプション--library-pathを使用して指定されたライブラリ検索パスで検索されます（[ライブラリ](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#libraries)を参照）。 |
| -D | 前処理時で使用するマクロを定義します。defは、<`name`>のみまたは<`name`>=<`definition`>のいずれで指定します。<br> name : 名前をマクロとして事前に定義します。<br> name = definition: ``#define``ディレクティブの変更フェーズのようにトークン化され処理されます。 |
| -U | 前処理またはコンパイル中に既存のマクロを未定義にします。 |
| -I | インクルード検索パスを指定します。 |
| -isystem | システムインクルード検索パスを指定します。 |
| -L | ライブラリ検索パスを指定します。 |
| -odir | 出力ファイルのディレクトリを指定します。 |
| -MF | 依存関係出力ファイルを指定します。 |
| -ccbin | コンパイラの実行可能ファイルが存在するディレクトリを指定します。|
| -arbin | --libで静的ライブラリを作成するために使用するアーカイバツールのパスを指定します。 |
| -ptx | すべての.cu入力ファイルをデバイス専用の.ptxファイルにコンパイルします。ホストコードを破棄。 |
| -arch | GPU仮想アーキテクチャのクラスの名前を指定します。実際のアーキテクチャを指定する場合最も近い仮想アーキテクチャをしようします。サポートされている仮想アーキテクチャのリストについては、[仮想アーキテクチャ機能リスト](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)を、サポートされている実際のアーキテクチャのリストについては、[GPU機能リスト](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)を参照してください。 |
| -code | PTXをアセンブルおよび最適化するNVIDIA GPUの名前を指定します。サポートされている仮想アーキテクチャのリストについては、[仮想アーキテクチャ機能リスト](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)を、サポートされている実際のアーキテクチャのリストについては、[GPU機能リスト](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)を参照してください。 |
| -Xcompiler | コンパイラー/プリプロセッサーに直接オプションを指定します。 | 
...etc


## GPU アーキテクチャ
| Fermi | Kepler | Mawell | Pascal | Volta | Turing | Ampere | Hopper |
|-|-|-|-|-|-|-|-|
| sm_20 | sm_30 | sm_50 | sm_60 | sm_70 | sm_75 | sm_80 | sm_85 |
| | sm_35 | sm_52 | sm_61 | sm_72 | | | |
| | sm_37 | sm_53 | sm_62 | | | | |

## SM (Streaming Multiprocessor)
- CUDA core
- shared memory / L1 cache
- register
- load / store unit
- SFU(Special Function Unit)
- Warp scheduler