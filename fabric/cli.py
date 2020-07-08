import typer
import uvicorn

from fabric.idm_factory import main as idm_factory
from fabric.utility import parse_file
from fabric.api import main as api_main

app = typer.Typer()
idm_app = typer.Typer()
app.add_typer(idm_app, name="idm")
api_app = typer.Typer()
app.add_typer(api_app, name="api")

"""
Infectious Disease Modeling application
"""
@idm_app.command("pipeline")
def idm_pipeline(pipeline: str,
                 config: str,
                 run_mode: str = "local",
                 start_date: str = None,
                 min_cases: int = None,
                 metric: str = "CONFIRMED",
                 src_from: str = "db",
                 base_path: str = None):

    pipeline_word = typer.style(pipeline, fg=typer.colors.GREEN, bold=True)
    typer.echo("Running pipeline: " + pipeline_word)
    idm_factory(pipeline, config, run_mode, start_date, metric, min_cases, src_from, base_path)

"""
API application
"""
@api_app.command("start")
def run_api(config: str, port: int = 8000, host: str = "0.0.0.0"):
    api_word = typer.style("api...", fg=typer.colors.GREEN, bold=True)
    typer.echo("Starting " + api_word)
    variables = parse_file(config, "API")
    api_model = api_main(variables)
    uvicorn.run(api_model, host=host, port=port)

if __name__ == "__main__":
    app()
