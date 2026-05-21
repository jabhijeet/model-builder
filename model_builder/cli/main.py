from typing import Optional
import typer
from .. import __version__
from .commands.init import init_command
from .commands.run import run_command
from .commands.status import status_command
from .commands.approve import approve_command, skip_command, retry_command
from .commands.runs import runs_command, compare_command
from .commands.ui import ui_command
from .commands.logs import logs_command
from .commands.export import export_command
from .commands.deploy import deploy_command
from .commands.tune import tune_command
from .commands.features import app as features_app
from .commands.models import app as models_app

app = typer.Typer(
    name="model-builder",
    help="Privacy-first local ML model builder",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"model-builder {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    pass


app.command("init")(init_command)
app.command("run")(run_command)
app.command("status")(status_command)
app.command("approve")(approve_command)
app.command("skip")(skip_command)
app.command("retry")(retry_command)
app.command("runs")(runs_command)
app.command("compare")(compare_command)
app.command("ui")(ui_command)
app.command("logs")(logs_command)
app.command("export")(export_command)
app.command("deploy")(deploy_command)
app.command("tune")(tune_command)
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")

if __name__ == "__main__":
    app()
