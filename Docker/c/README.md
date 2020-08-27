# CUDAのテスト

## CUDAの実行
```
nvcc -arch <最適化アーキテクチャ> <ソースファイル> -o <出力ファイル名>
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