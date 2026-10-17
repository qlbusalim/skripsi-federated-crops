"""Factory for the recurrent regression models in this repository."""

from __future__ import annotations

from typing import Any

from .recurrent import BiLSTM, GRU, LSTM, RNN


MODEL_TYPES = {"rnn": RNN, "lstm": LSTM, "gru": GRU, "bilstm": BiLSTM}


def create_model(config: dict[str, Any], **overrides: Any):
	"""Create a configured recurrent regression model."""
	model_config = config.get("model", config)
	model_name = str(model_config.get("name", "lstm")).lower()
	if model_name not in MODEL_TYPES:
		raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(MODEL_TYPES)}")
	parameters = {
		"input_size": model_config.get("input_size", 1),
		"hidden_size": model_config.get("hidden_size", 32),
		"num_layers": model_config.get("num_layers", 1),
		"output_size": model_config.get("output_size", 1),
	}
	parameters.update(overrides)
	return MODEL_TYPES[model_name](**parameters)


__all__ = ["MODEL_TYPES", "create_model"]
