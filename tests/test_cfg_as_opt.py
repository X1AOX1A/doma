import json
import click
from click.testing import CliRunner
from pydantic import BaseModel, Field

import doma.utils as utils


def test_cmd_executes_with_defaults_and_required(monkeypatch):
    # Isolate CLI group per test
    test_cli = click.Group()

    class TestConfig(BaseModel):
        a: int = Field(description="A number")
        b_value: float = Field(default=1.5, description="B value")
    @test_cli.command()
    @utils.cfg_as_opt(TestConfig)
    def handle(config: TestConfig):
        click.echo(json.dumps(config.model_dump()))

    # Discover the registered command name (wrapper function name)
    assert len(test_cli.commands) == 1
    cmd_name = next(iter(test_cli.commands))

    runner = CliRunner()
    result = runner.invoke(test_cli, [cmd_name, "--a", "3"])  # b_value uses default

    assert result.exit_code == 0
    payload = json.loads(result.output.strip())
    assert payload == {"a": 3, "b_value": 1.5}


def test_underscore_to_dash_option_and_overrides(monkeypatch):
    test_cli = click.Group()

    class TestConfig(BaseModel):
        a: int = Field(description="A number")
        b_value: float = Field(default=1.5, description="B value")

    @test_cli.command()
    @utils.cfg_as_opt(TestConfig)
    def handle(config: TestConfig):
        click.echo(json.dumps(config.model_dump()))

    cmd_name = next(iter(test_cli.commands))

    runner = CliRunner()
    result = runner.invoke(
        test_cli,
        [cmd_name, "--a", "2", "--b-value", "4.2"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip())
    assert payload == {"a": 2, "b_value": 4.2}

    # Help text should include option descriptions
    help_result = runner.invoke(test_cli, [cmd_name, "--help"]) 
    assert help_result.exit_code == 0
    assert "A number" in help_result.stdout
    assert "B value" in help_result.stdout


def test_pydantic_validation_error_propagates(monkeypatch):
    test_cli = click.Group()

    class PositiveOnly(BaseModel):
        a: int = Field(gt=0, description="Positive integer")

    @test_cli.command()
    @utils.cfg_as_opt(PositiveOnly)
    def handle(config: PositiveOnly):
        click.echo(json.dumps(config.model_dump()))

    cmd_name = next(iter(test_cli.commands))

    runner = CliRunner()
    # Click accepts 0 for --a (type=int), but Pydantic should reject it (gt=0)
    result = runner.invoke(test_cli, [cmd_name, "--a", "0"])
    assert result.exit_code != 0
    assert result.exception is not None
    # Missing required option also triggers validation error
    result_missing = runner.invoke(test_cli, [cmd_name])
    assert result_missing.exit_code != 0
    assert result_missing.exception is not None

