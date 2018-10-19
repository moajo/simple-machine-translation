# simple seq2seq translation sample

seq2seq モデルによる日英翻訳のサンプル実装。

# requires

- pytorch
- numpy
- tqdm

# usage

データは [small_parallel_enja](https://github.com/odashi/small_parallel_enja)を推奨

download スクリプトを用意しています。

```sh
./download.sh
```

以下のように学習を始めます。

```sh
./train.py
```

ソースを少し書き換えて、学習済みモデルをロードしたりパラメータを変更したりできます。
GPU も利用できます。
