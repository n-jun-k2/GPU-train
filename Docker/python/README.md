# GPU環境下のPython開発のメモ



## scikit-learnの並列処理のバージョン対応 <[詳細](https://scikit-learn.org/stable/modules/computing.html#openmp-based-parallelism)>
以前までの``n_jobs``で並列数を指定していましたが現在使用しているバージョンでは以下のように対応すること。
```
> OMP_NUM_THREADS=4 python script.py
```