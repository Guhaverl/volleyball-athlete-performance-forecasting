class VolleyForecastError(Exception):
    """Base exception for repository-specific failures."""


class DataContractError(VolleyForecastError):
    """Raised when an input table violates the canonical data contract."""


class SourceAdapterError(VolleyForecastError):
    """Raised when a source response cannot be converted to canonical rows."""


class TensorFlowUnavailableError(VolleyForecastError):
    """Raised when an ML command is used without the TensorFlow extra."""
