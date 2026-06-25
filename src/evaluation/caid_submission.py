from __future__ import annotations

"""Compatibility exports for assessor-facing CAID submission writers."""

from src.evaluation.caid_io import SUBMISSION_TASKS, write_caid_file, write_submission_bundle

__all__ = ["SUBMISSION_TASKS", "write_caid_file", "write_submission_bundle"]
