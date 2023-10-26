# GPU環境下のPython開発のメモ



## scikit-learnの並列処理のバージョン対応 <[詳細](https://scikit-learn.org/stable/modules/computing.html#openmp-based-parallelism)>
以前までの``n_jobs``で並列数を指定していましたが現在使用しているバージョンでは以下のように対応すること。
```
> OMP_NUM_THREADS=4 python script.py
```

## hugging face login

[Hugging Faceの設定](https://huggingface.co/settings/tokens)から"Access Tokens"を選択し
User Access Tokensを作成します。

作成後、CLIでトークンを指定

```bash
> huggingface-cli login --token <TOKEN>
```

ログイン後、PythonのAPIで操作が可能。