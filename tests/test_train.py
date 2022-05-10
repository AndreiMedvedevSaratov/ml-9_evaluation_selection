from click.testing import CliRunner  # type: ignore
from evaluation_selection_project.train import train  # type: ignore
import pytest  # type: ignore
from click.testing import CliRunner
import pytest

from evaluation_selection_project.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(
    runner: CliRunner
) -> None:
    """It fails when test split ratio is greater than 10"""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            33,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output


def test_error_for_invalid_test_split_ratio(runner: CliRunner) -> None:
    """It fails when test split ratio is less than 0."""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            "-10",
        ],
    )
    assert result.exit_code == 1
