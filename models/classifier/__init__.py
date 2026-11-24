"""Discriminative MLP and the decoder trained on its frozen features."""

from models.classifier.frozen_decoder import FrozenDecoder
from models.classifier.mlp_classifier import MLPClassifier

__all__ = ["FrozenDecoder", "MLPClassifier"]
