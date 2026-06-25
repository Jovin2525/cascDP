"""Evaluation metrics for cascDP model."""

__all__ = ['MetricsCalculator']

def __getattr__(name):
	if name == 'MetricsCalculator':
		from .metrics import MetricsCalculator

		return MetricsCalculator

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

