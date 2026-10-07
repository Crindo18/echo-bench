"""Evaluation: few-shot enrollment metrics, cross-validation protocol,
latency/RAM profiling, and the model-selection scorecard."""
from .benchmark import evaluate, extract_embeddings, accuracy, macro_f1, print_report
from .latency import profile_onnx
from .protocol import cross_validate, print_accuracy
from .sweep import metatrain_eval_sweep, print_sweep
from .scorecard import (ModelCard, card_from_measurements, build_scorecard,
                        print_scorecard, weights_without_severity, DEFAULT_WEIGHTS)

__all__ = ["evaluate", "extract_embeddings", "accuracy", "macro_f1",
           "print_report", "profile_onnx",
           "cross_validate", "print_accuracy",
           "metatrain_eval_sweep", "print_sweep",
           "ModelCard", "card_from_measurements", "build_scorecard",
           "print_scorecard", "weights_without_severity", "DEFAULT_WEIGHTS"]
