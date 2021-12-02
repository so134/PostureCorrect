# 姿勢を検出してアラートを表示するためのアプリケーション

## 動作環境
```
OS: macOS Catalina

CPU: 2.6 GHz 6コア Intel Core i7

メモリ： 32 GB 2667 MHz DDR4

Python : 3.6.7

TensorFlow : 1.15.0

OpenCv : 4.5.4
```
## Quick Start

1. 自身のlocalに tf-openposeの環境を作成してください。
https://github.com/jiajunhua/ildoonet-tf-pose-estimation#install
ildoonet-tf-pose-estimationディレクトリ内の
`run.py` と同じ階層に本リポジトリの `pose_monitor.py` を配置し、

```
python ildoonet-tf-pose-estimation/pose_monitor.py
```

で実行
