import typer, logging
from rich.console import Console
from rich.table import Table
import neuralforge as nf
from neuralforge.spec import NeuralForgeSpec, TrainingConfig, ExportFormat
app = typer.Typer(name="neuralforge", help="NeuralForge v2.0", no_args_is_help=True)
console = Console()

@app.command()
def create(description: str = typer.Argument(...), name: str = typer.Option(None, "--name", "-n"),
           save: bool = typer.Option(True, "--save/--no-save"), output_dir: str = typer.Option("./neuralforge_models", "--output-dir", "-o")):
    spec = NeuralForgeSpec.from_description(description)
    if name: spec.name = name
    model = nf.create_model(spec)
    table = Table(title=f"Model: {spec.name}", show_header=False, border_style="cyan")
    table.add_row("Architecture", spec.architecture.family.value); table.add_row("Parameters", f"{model.count_parameters():,}")
    table.add_row("Config Hash", spec.config_hash()); console.print(table)
    if save:
        from pathlib import Path; out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{spec.name}_spec.yaml", "w") as f: f.write(spec.to_yaml())
        console.print(f"Spec saved to {out / f'{spec.name}_spec.yaml'}")

@app.command()
def info():
    import torch
    table = Table(show_header=False, border_style="cyan")
    table.add_row("Version", nf.__version__); table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA", str(torch.cuda.is_available())); console.print(table)

@app.command()
def list_models():
    models = nf.list_models()
    if not models: console.print("No models registered."); return
    table = Table(title="Registered Models", border_style="cyan")
    table.add_column("Name", style="bold"); table.add_column("Version"); table.add_column("Hash")
    for m in models: table.add_row(m["name"], m["version"], m.get("config_hash", "N/A")[:12])
    console.print(table)

if __name__ == "__main__": app()
