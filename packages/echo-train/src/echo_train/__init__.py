"""echo_train: build, export and quantize the encoder (desktop only, own lockfile, ADR-6).

config.py       reads configs/models/*.yaml
build_model.py  E-Branchformer encoder + masked mean pooling -> EmbeddingNet
export_onnx.py  ONNX FP32 export and the multi-length parity test (gate G2a)
quantize.py     preprocess -> dynamic | static QDQ INT8 versions
artifacts.py    writes artifacts/models/<name>/<sha8>/ folders with model cards
cli.py          the `echo-train` command
"""

__version__ = "0.1.0"
