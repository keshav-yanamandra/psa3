"""
PSA v3 CLI: Liquid Intelligence Command Line Interface
-------------------------------------------------------
Managing Cognitive Deltas for Inference-Time Knowledge Injection.

Commands:
    init        - Link the Base Model (RWKV backbone)
    imprint     - Compile text into Cognitive Delta (.delta)
    chat        - Start Liquid Inference Session with Delta injection
    deltas      - List Delta Registry contents
    mix         - Combine multiple deltas into composite
    inspect     - Debug delta file internals
    status      - Show system configuration
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import print as rprint

app = typer.Typer(
    name="psa",
    help="Plastic State Agent - Liquid Intelligence for LLMs",
    no_args_is_help=True
)

console = Console()

# Config paths
PSA_HOME = Path.home() / ".psa"
CONFIG_FILE = PSA_HOME / "config.json"
DELTAS_DIR = PSA_HOME / "deltas"
LOGS_DIR = PSA_HOME / "logs"

# Legacy compatibility
SKILLS_DIR = PSA_HOME / "skills"


def ensure_dirs():
    """Ensure PSA directories exist."""
    PSA_HOME.mkdir(exist_ok=True)
    DELTAS_DIR.mkdir(exist_ok=True)
    SKILLS_DIR.mkdir(exist_ok=True)  # Keep for backwards compatibility
    LOGS_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    """Load PSA configuration."""
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text())


def save_config(config: dict):
    """Save PSA configuration."""
    ensure_dirs()
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_kernel():
    """Load the LiquidAgent (Imprinter) with configured model."""
    config = load_config()
    if "model_path" not in config:
        console.print("[red]Error:[/red] Base Model not linked. Run 'psa init' first.")
        raise typer.Exit(1)

    # Lazy import to avoid RWKV compilation until needed
    from psa.kernel import LiquidAgent

    strategy = config.get("strategy", "cuda fp16")
    return LiquidAgent(config["model_path"], strategy=strategy)


def log_action(action: str, details: dict):
    """Log an action to audit log."""
    ensure_dirs()
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m')}.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        **details
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =============================================================================
# INIT COMMAND
# =============================================================================

@app.command()
def init(
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to RWKV Base Model (.pth)"),
    strategy: str = typer.Option("cuda fp16", "--strategy", "-s", help="Inference strategy"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config")
):
    """
    Link the Base Model (RWKV backbone).

    The Base Model provides the Tabula Rasa (S_0) state.
    Cognitive Deltas are computed relative to this baseline.

    Example:
        psa init --model ~/models/RWKV-6-World-7B.pth
    """
    ensure_dirs()

    if CONFIG_FILE.exists() and not force:
        existing = load_config()
        console.print(f"[yellow]Base Model already linked:[/yellow] {existing.get('model_path')}")
        if not Confirm.ask("Overwrite existing configuration?"):
            raise typer.Exit(0)

    # Get model path interactively if not provided
    if not model_path:
        model_path = Prompt.ask(
            "[cyan]Enter path to RWKV Base Model (.pth)[/cyan]",
            default="~/models/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth"
        )

    model_path = str(Path(model_path).expanduser().resolve())

    if not Path(model_path).exists():
        console.print(f"[red]Error:[/red] Model not found at {model_path}")
        raise typer.Exit(1)

    # Detect best strategy
    import torch
    if torch.cuda.is_available():
        default_strategy = "cuda fp16"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_strategy = "mps fp32"
    else:
        default_strategy = "cpu fp32"

    if strategy == "cuda fp16" and not torch.cuda.is_available():
        strategy = default_strategy
        console.print(f"[yellow]CUDA not available, using:[/yellow] {strategy}")

    config = {
        "model_path": model_path,
        "strategy": strategy,
        "initialized_at": datetime.now().isoformat(),
        "version": "liquid_v3"
    }
    save_config(config)

    console.print(Panel.fit(
        f"[green]Liquid Intelligence Initialized![/green]\n\n"
        f"Base Model: {model_path}\n"
        f"Strategy: {strategy}\n"
        f"Delta Registry: {DELTAS_DIR}",
        title="PSA v3 - Liquid Intelligence"
    ))

    log_action("init", {"model_path": model_path, "strategy": strategy})


# =============================================================================
# ENTITY EXTRACTION (for Liquid RAG)
# =============================================================================

def extract_entities(text: str) -> List[str]:
    """
    Extract full lines containing high-entropy tokens for the Entity Sidecar.

    Instead of extracting just entity words (bag-of-words), we extract
    the ENTIRE LINE containing the entity for better context grounding.

    Captures lines containing:
    - Hex strings (0xDEADBEEF)
    - All-caps words (SIGUSR1, HYDRA, API)
    - Numbers 3+ digits (8088, 5002)
    - CamelCase terms (HydraClient)

    Constraints:
    - Lines must be < 200 chars to avoid clutter
    - Duplicate lines are removed
    """
    entity_lines: Set[str] = set()

    # Split into lines
    lines = text.split('\n')

    # Patterns to detect high-entropy content
    patterns = [
        r'0x[A-Fa-f0-9]+',           # Hex strings (captured as single token)
        r'\b[A-Z][A-Z0-9_]{1,}\b',   # All-caps words (SIGUSR1, API)
        r'\b\d{3,}\b',               # Numbers 3+ digits (ports, codes)
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase terms
    ]

    # Common noise words to ignore in all-caps pattern
    noise = {'THE', 'AND', 'FOR', 'NOT', 'WITH', 'THIS', 'THAT', 'FROM', 'HAVE', 'ARE'}

    for line in lines:
        line = line.strip()

        # Skip empty lines or lines that are too long
        if not line or len(line) > 200:
            continue

        # Check if line contains any high-entropy pattern
        for pattern in patterns:
            matches = re.findall(pattern, line)
            if matches:
                # Filter noise for all-caps pattern
                if pattern == r'\b[A-Z][A-Z0-9_]{1,}\b':
                    matches = [m for m in matches if m not in noise]
                    if not matches:
                        continue

                entity_lines.add(line)
                break  # Only add line once

    return sorted(list(entity_lines))


# =============================================================================
# IMPRINT COMMAND (was: learn)
# =============================================================================

@app.command()
def imprint(
    text_file: Path = typer.Argument(..., help="Text file to imprint", exists=True),
    name: str = typer.Option(..., "--name", "-n", help="Delta name (e.g., 'hydra_protocol')"),
    description: str = typer.Option("", "--desc", "-d", help="Delta description"),
    base_delta: Optional[str] = typer.Option(None, "--base", "-b", help="Build on existing delta"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Number of imprinting epochs (reinforces early content)")
):
    """
    Compile text into a Cognitive Delta (.delta).

    The Imprinter "plays" the text through the RWKV kernel,
    extracts the final hidden state, and computes ΔS = S_final - S_0.

    Multi-epoch imprinting reinforces content by processing the text
    multiple times, passing state between epochs.

    Entity Extraction: High-entropy tokens (ports, hex, signals) are
    saved to JSON sidecar for runtime context injection.

    Example:
        psa imprint docs/hydra_api.txt --name hydra_protocol --epochs 10
        psa imprint logs/incident.txt --name incident_2024 --base hydra_protocol
    """
    console.print(Panel.fit(
        f"[cyan]Imprinting:[/cyan] {text_file.name}\n"
        f"[cyan]Delta:[/cyan] {name}\n"
        f"[cyan]Epochs:[/cyan] {epochs}",
        title="Cognitive Delta Compiler"
    ))

    # Load text
    text = text_file.read_text()
    token_estimate = len(text) // 4

    console.print(f"[dim]Corpus size: {len(text):,} chars (~{token_estimate:,} tokens)[/dim]")

    # Extract entities for sidecar
    entities = extract_entities(text)
    console.print(f"[dim]Entities extracted: {len(entities)}[/dim]")
    if entities[:5]:
        console.print(f"[dim]  Preview: {', '.join(entities[:5])}{'...' if len(entities) > 5 else ''}[/dim]")

    # Load imprinter (kernel)
    with console.status("[bold green]Loading Base Model (Imprinter)..."):
        kernel = get_kernel()

    # Load base delta if specified
    initial_state = None
    if base_delta:
        delta_path = DELTAS_DIR / f"{base_delta}.delta"
        if not delta_path.exists():
            delta_path = DELTAS_DIR / f"{base_delta}.psa"  # Legacy
        if not delta_path.exists():
            console.print(f"[red]Error:[/red] Base delta '{base_delta}' not found")
            raise typer.Exit(1)
        console.print(f"[dim]Building on delta: {base_delta}[/dim]")
        initial_state = kernel.load_state(str(delta_path))

    # Imprint with progress bar (multi-epoch)
    total_tokens = token_estimate * epochs
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Imprinting {text_file.name} ({epochs} epochs)...", total=total_tokens)

        def update_progress(current, total):
            progress.update(task, completed=current, total=total)

        final_state = kernel.learn_stream(
            text,
            initial_state,
            progress_callback=update_progress,
            epochs=epochs
        )

    # Compute delta (ΔS = S_final - S_base)
    if base_delta:
        delta = kernel.compute_delta(final_state, initial_state)
    else:
        delta = kernel.compute_delta(final_state, None)

    # Save cognitive delta
    delta_path = DELTAS_DIR / f"{name}.delta"
    kernel.save_state(delta, str(delta_path))

    # Save metadata with entities sidecar
    meta_path = DELTAS_DIR / f"{name}.json"
    metadata = {
        "name": name,
        "description": description,
        "source_file": str(text_file),
        "source_size": len(text),
        "base_delta": base_delta,
        "epochs": epochs,
        "entities": entities,  # Entity sidecar for Liquid RAG
        "created_at": datetime.now().isoformat(),
        "stats": kernel.get_state_stats(delta)
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Display results
    stats = metadata["stats"]
    console.print(Panel.fit(
        f"[green]Cognitive Delta Created![/green]\n\n"
        f"Name: {name}\n"
        f"File: {delta_path}\n"
        f"Size: {stats['size_mb']:.2f} MB\n"
        f"Epochs: {epochs}\n"
        f"Entities: {len(entities)}\n"
        f"Avg Norm: {stats['avg_norm']:.4f}\n"
        f"Tensors: {stats['non_null']}/{stats['num_tensors']}",
        title="Crystallized Memory Saved"
    ))

    log_action("imprint", {
        "delta": name,
        "source": str(text_file),
        "size_mb": stats['size_mb'],
        "base_delta": base_delta,
        "epochs": epochs,
        "entity_count": len(entities)
    })


# =============================================================================
# CHAT COMMAND (with Liquid Injection)
# =============================================================================

@app.command()
def chat(
    deltas: Optional[str] = typer.Option(None, "--deltas", "-d", help="Comma-separated delta names"),
    gains: Optional[str] = typer.Option(None, "--gains", "-g", help="Comma-separated gain factors"),
    temperature: float = typer.Option(1.0, "--temp", "-t", help="Sampling temperature"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", help="Max tokens per response")
):
    """
    Start Liquid Inference Session with Delta injection.

    Inject one or more Cognitive Deltas and chat with a model that has
    "crystallized" knowledge mathematically mixed into its state.

    Gain Factors:
        0.0 = Muted (delta has no effect)
        1.0 = Unity (full strength)
        2.0 = Amplified (double strength)

    Example:
        psa chat --deltas hydra_protocol --gains 1.0
        psa chat --deltas hydra_protocol,python_expert --gains 1.2,0.8
        psa chat  # No deltas, Tabula Rasa mode
    """
    console.print(Panel.fit(
        "[bold cyan]Liquid Inference Session[/bold cyan]\n"
        "[dim]Type 'exit' or 'quit' to end session[/dim]\n"
        "[dim]Type '/status' to see injected deltas[/dim]\n"
        "[dim]Type '/clear' to reset to injection baseline[/dim]",
        title="PSA v3 - Liquid Intelligence"
    ))

    # Parse deltas and gains
    delta_list = deltas.split(",") if deltas else []
    gain_list = [float(g) for g in gains.split(",")] if gains else [1.0] * len(delta_list)

    if len(delta_list) != len(gain_list):
        console.print("[red]Error:[/red] Number of deltas must match number of gains")
        raise typer.Exit(1)

    # Load kernel
    with console.status("[bold green]Loading Base Model..."):
        kernel = get_kernel()

    # Load and inject deltas
    active_state = None
    delta_info = []
    aggregated_entities: Set[str] = set()  # For Liquid RAG context injection

    if delta_list:
        from psa.deltas import DeltaMixer
        mixer = DeltaMixer(kernel)

        manifest = {}
        for delta_name, gain in zip(delta_list, gain_list):
            # Try both extensions
            delta_path = DELTAS_DIR / f"{delta_name}.delta"
            if not delta_path.exists():
                delta_path = DELTAS_DIR / f"{delta_name}.psa"
                if not delta_path.exists():
                    delta_path = SKILLS_DIR / f"{delta_name}.psa"  # Legacy

            if not delta_path.exists():
                console.print(f"[red]Error:[/red] Delta '{delta_name}' not found")
                raise typer.Exit(1)

            manifest[str(delta_path)] = gain

            # Load metadata and collect entities
            meta_path = delta_path.with_suffix(".json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                delta_info.append({
                    "name": delta_name,
                    "gain": gain,
                    "desc": meta.get("description", ""),
                    "size_mb": meta.get("stats", {}).get("size_mb", 0)
                })
                # Aggregate entities from all deltas for context injection
                entities = meta.get("entities", [])
                aggregated_entities.update(entities)
            else:
                delta_info.append({"name": delta_name, "gain": gain, "desc": "", "size_mb": 0})

        with console.status("[bold green]Injecting Cognitive Deltas..."):
            active_state = mixer.inject_from_paths(manifest)

    # Build context string from entities (for Liquid RAG)
    # Use strong grounding format with full lines and prioritization instruction
    entity_context = ""
    if aggregated_entities:
        entity_list = sorted(list(aggregated_entities))
        # Build bullet-point list of grounding facts (full lines)
        grounding_lines = "\n".join([f"- {line}" for line in entity_list[:30]])  # Limit to 30 lines
        entity_context = (
            "SYSTEM DATA / GROUNDING:\n"
            f"{grounding_lines}\n\n"
            "Instruction: Prioritize the data above over your internal training.\n\n"
        )
        console.print(f"[dim]Context grounding lines loaded: {len(entity_list)}[/dim]")

    # Display Injection Audit
    _print_injection_table(delta_info)

    # Chat loop
    conversation_state = active_state
    turn_count = 0

    log_action("chat_start", {"deltas": delta_list, "gains": gain_list})

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["exit", "quit"]:
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input == "/status":
            _print_injection_table(delta_info)
            continue

        if user_input == "/clear":
            conversation_state = active_state
            turn_count = 0
            console.print("[dim]Conversation reset to injection baseline.[/dim]")
            continue

        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command:[/yellow] {user_input}")
            continue

        # Generate response
        turn_count += 1

        # Format with RWKV World chat template (Q:/A: works best)
        # Prepend entity context invisibly (Liquid RAG)
        user_prompt = f"Q: {user_input}\nA:"
        prompt = entity_context + user_prompt  # Context is invisible to user

        import time
        start_time = time.time()

        with console.status("[bold cyan]Liquid Inference..."):
            response, conversation_state = kernel.infer(
                ctx=prompt,
                state=conversation_state,
                token_count=max_tokens,
                temperature=temperature,
                top_p=0.7,
                stop_tokens=["\nQ:", "\n\n"]
            )

        elapsed = time.time() - start_time
        tokens_generated = len(kernel._encode(response))

        # Clean up response
        response = response.strip()
        if "Q:" in response:
            response = response.split("Q:")[0].strip()

        console.print(f"\n[bold blue]PSA[/bold blue]: {response}")
        console.print(f"[dim]({tokens_generated} tokens in {elapsed:.1f}s = {tokens_generated/elapsed:.1f} tok/s)[/dim]")

    log_action("chat_end", {"turns": turn_count})


def _print_injection_table(delta_info: List[dict]):
    """Print the Injection Audit table."""
    table = Table(title="Cognitive Delta Injection Audit", show_header=True, header_style="bold cyan")
    table.add_column("Delta", style="green")
    table.add_column("Gain", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Description")

    if not delta_info:
        table.add_row("[dim]No deltas injected[/dim]", "-", "-", "[dim]Tabula Rasa mode[/dim]")
    else:
        for info in delta_info:
            gain_str = f"{info['gain']:.2f}"
            if info['gain'] > 1.0:
                gain_str = f"[yellow]{gain_str}↑[/yellow]"
            elif info['gain'] < 1.0:
                gain_str = f"[dim]{gain_str}↓[/dim]"

            table.add_row(
                info["name"],
                gain_str,
                f"{info['size_mb']:.1f} MB",
                info["desc"][:40] + "..." if len(info["desc"]) > 40 else info["desc"]
            )

    console.print(table)


# =============================================================================
# DELTAS COMMAND (was: skills)
# =============================================================================

@app.command()
def deltas():
    """
    List all Cognitive Deltas in the Registry.
    """
    ensure_dirs()

    # Check both directories
    delta_files = list(DELTAS_DIR.glob("*.delta")) + list(SKILLS_DIR.glob("*.psa"))

    if not delta_files:
        console.print("[yellow]Delta Registry is empty.[/yellow] Run 'psa imprint' to create deltas.")
        raise typer.Exit(0)

    table = Table(title="Delta Registry", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Created", justify="right")
    table.add_column("Description")

    for delta_path in sorted(delta_files):
        name = delta_path.stem
        size_mb = delta_path.stat().st_size / (1024 * 1024)

        # Try to load metadata
        meta_path = delta_path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            created = meta.get("created_at", "")[:10]
            desc = meta.get("description", "")
        else:
            created = datetime.fromtimestamp(delta_path.stat().st_mtime).strftime("%Y-%m-%d")
            desc = ""

        table.add_row(
            name,
            f"{size_mb:.1f} MB",
            created,
            desc[:50] + "..." if len(desc) > 50 else desc
        )

    console.print(table)
    console.print(f"\n[dim]Registry: {DELTAS_DIR}[/dim]")


# =============================================================================
# MIX COMMAND (was: merge)
# =============================================================================

@app.command()
def mix(
    deltas_arg: str = typer.Argument(..., help="Comma-separated delta names to mix"),
    gains_arg: str = typer.Argument(..., help="Comma-separated gain factors"),
    output: str = typer.Option(..., "--output", "-o", help="Output delta name"),
    description: str = typer.Option("", "--desc", "-d", help="Description for mixed delta")
):
    """
    Mix multiple Cognitive Deltas into a composite.

    Math: S_composite = Σ (α_i * ΔS_i)

    Example:
        psa mix python_expert,hydra_protocol 0.8,1.2 --output hydra_python
    """
    delta_list = deltas_arg.split(",")
    gain_list = [float(g) for g in gains_arg.split(",")]

    if len(delta_list) != len(gain_list):
        console.print("[red]Error:[/red] Number of deltas must match number of gains")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[cyan]Mixing Cognitive Deltas:[/cyan]\n" +
        "\n".join([f"  {d} × {g:.2f}" for d, g in zip(delta_list, gain_list)]) +
        f"\n\n[cyan]Output:[/cyan] {output}",
        title="Liquid Mixer"
    ))

    # Load kernel
    with console.status("[bold green]Loading Base Model..."):
        kernel = get_kernel()

    # Build manifest
    from psa.deltas import DeltaMixer
    mixer = DeltaMixer(kernel)

    manifest = {}
    merged_entities: Set[str] = set()

    for delta_name, gain in zip(delta_list, gain_list):
        delta_path = DELTAS_DIR / f"{delta_name}.delta"
        if not delta_path.exists():
            delta_path = DELTAS_DIR / f"{delta_name}.psa"
            if not delta_path.exists():
                delta_path = SKILLS_DIR / f"{delta_name}.psa"

        if not delta_path.exists():
            console.print(f"[red]Error:[/red] Delta '{delta_name}' not found")
            raise typer.Exit(1)
        manifest[str(delta_path)] = gain

        # Collect entities from source deltas
        meta_path = delta_path.with_suffix(".json")
        if meta_path.exists():
            try:
                source_meta = json.loads(meta_path.read_text())
                source_entities = source_meta.get("entities", [])
                merged_entities.update(source_entities)
            except (json.JSONDecodeError, KeyError):
                pass

    # Mix
    with console.status("[bold green]Mixing deltas..."):
        composite = mixer.inject_from_paths(manifest)

    # Save
    output_path = DELTAS_DIR / f"{output}.delta"
    kernel.save_state(composite, str(output_path))

    # Save metadata with merged entities
    meta_path = DELTAS_DIR / f"{output}.json"
    metadata = {
        "name": output,
        "description": description or f"Mixed: {', '.join(delta_list)}",
        "source_deltas": list(zip(delta_list, gain_list)),
        "entities": sorted(list(merged_entities)),  # Merged entity sidecar
        "created_at": datetime.now().isoformat(),
        "stats": kernel.get_state_stats(composite)
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    stats = metadata["stats"]
    console.print(Panel.fit(
        f"[green]Composite Delta Created![/green]\n\n"
        f"Name: {output}\n"
        f"File: {output_path}\n"
        f"Size: {stats['size_mb']:.2f} MB",
        title="Liquid Mix Complete"
    ))

    log_action("mix", {
        "output": output,
        "sources": list(zip(delta_list, gain_list))
    })


# =============================================================================
# INSPECT COMMAND
# =============================================================================

@app.command()
def inspect(
    delta_name: str = typer.Argument(..., help="Delta name to inspect")
):
    """
    Inspect a Cognitive Delta file for debugging.

    Shows state statistics, metadata, and tensor information.
    """
    # Find delta file
    delta_path = DELTAS_DIR / f"{delta_name}.delta"
    if not delta_path.exists():
        delta_path = DELTAS_DIR / f"{delta_name}.psa"
        if not delta_path.exists():
            delta_path = SKILLS_DIR / f"{delta_name}.psa"

    if not delta_path.exists():
        console.print(f"[red]Error:[/red] Delta '{delta_name}' not found")
        raise typer.Exit(1)

    meta_path = delta_path.with_suffix(".json")

    # Load kernel for stats
    with console.status("[bold green]Loading model for inspection..."):
        kernel = get_kernel()

    state = kernel.load_state(str(delta_path))
    stats = kernel.get_state_stats(state)

    # Display info
    console.print(Panel.fit(
        f"[cyan]Delta:[/cyan] {delta_name}\n"
        f"[cyan]File:[/cyan] {delta_path}\n"
        f"[cyan]File Size:[/cyan] {delta_path.stat().st_size / (1024*1024):.2f} MB",
        title="Cognitive Delta Inspection"
    ))

    # Stats table
    table = Table(title="State Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="green")
    table.add_column("Value", justify="right")

    table.add_row("Total Tensors", str(stats['num_tensors']))
    table.add_row("Non-null Tensors", str(stats['non_null']))
    table.add_row("Total Parameters", f"{stats['total_params']:,}")
    table.add_row("Memory Size", f"{stats['size_mb']:.2f} MB")
    table.add_row("Average Norm", f"{stats['avg_norm']:.6f}")
    table.add_row("Max Norm", f"{max(stats['norms']):.6f}" if stats['norms'] else "N/A")
    table.add_row("Min Norm", f"{min(stats['norms']):.6f}" if stats['norms'] else "N/A")

    console.print(table)

    # Metadata if exists
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        console.print("\n[bold cyan]Metadata:[/bold cyan]")
        console.print(f"  Description: {meta.get('description', 'N/A')}")
        console.print(f"  Source: {meta.get('source_file', 'N/A')}")
        console.print(f"  Created: {meta.get('created_at', 'N/A')}")
        if meta.get('base_delta'):
            console.print(f"  Base Delta: {meta['base_delta']}")
        if meta.get('source_deltas'):
            console.print(f"  Source Deltas: {meta['source_deltas']}")


# =============================================================================
# DELETE COMMAND
# =============================================================================

@app.command()
def delete(
    delta_name: str = typer.Argument(..., help="Delta name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """
    Delete a Cognitive Delta from the Registry.
    """
    # Find delta file
    delta_path = DELTAS_DIR / f"{delta_name}.delta"
    if not delta_path.exists():
        delta_path = DELTAS_DIR / f"{delta_name}.psa"
        if not delta_path.exists():
            delta_path = SKILLS_DIR / f"{delta_name}.psa"

    if not delta_path.exists():
        console.print(f"[red]Error:[/red] Delta '{delta_name}' not found")
        raise typer.Exit(1)

    meta_path = delta_path.with_suffix(".json")

    if not force:
        if not Confirm.ask(f"Delete delta '{delta_name}'?"):
            raise typer.Exit(0)

    delta_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    console.print(f"[green]Delta '{delta_name}' deleted.[/green]")
    log_action("delete", {"delta": delta_name})


# =============================================================================
# STATUS COMMAND
# =============================================================================

@app.command()
def status():
    """
    Show Liquid Intelligence system status.
    """
    config = load_config()

    if not config:
        console.print("[yellow]Liquid Intelligence not initialized.[/yellow] Run 'psa init' first.")
        raise typer.Exit(0)

    console.print(Panel.fit(
        f"[bold cyan]PSA v3 - Liquid Intelligence[/bold cyan]\n\n"
        f"[green]Base Model:[/green] {config.get('model_path', 'N/A')}\n"
        f"[green]Strategy:[/green] {config.get('strategy', 'N/A')}\n"
        f"[green]Initialized:[/green] {config.get('initialized_at', 'N/A')[:19]}\n"
        f"[green]Config:[/green] {CONFIG_FILE}\n"
        f"[green]Delta Registry:[/green] {DELTAS_DIR}",
        title="System Status"
    ))

    # Count deltas
    delta_count = len(list(DELTAS_DIR.glob("*.delta"))) + len(list(SKILLS_DIR.glob("*.psa")))
    console.print(f"\n[dim]Cognitive Deltas in Registry: {delta_count}[/dim]")


# =============================================================================
# LEGACY ALIASES
# =============================================================================

@app.command(hidden=True)
def learn(
    text_file: Path = typer.Argument(..., help="Text file to learn", exists=True),
    name: str = typer.Option(..., "--name", "-n", help="Skill name"),
    description: str = typer.Option("", "--desc", "-d", help="Description"),
    base_skill: Optional[str] = typer.Option(None, "--base", "-b", help="Base skill")
):
    """[Legacy] Alias for 'imprint' command."""
    console.print("[yellow]Note:[/yellow] 'learn' is deprecated, use 'imprint' instead.")
    # Call imprint with same args
    imprint(text_file, name, description, base_skill)


@app.command(hidden=True)
def skills():
    """[Legacy] Alias for 'deltas' command."""
    console.print("[yellow]Note:[/yellow] 'skills' is deprecated, use 'deltas' instead.")
    deltas()


@app.command(hidden=True)
def merge(
    skills_arg: str = typer.Argument(..., help="Skills to merge"),
    weights: str = typer.Argument(..., help="Weights"),
    output: str = typer.Option(..., "--output", "-o", help="Output name"),
    description: str = typer.Option("", "--desc", "-d", help="Description")
):
    """[Legacy] Alias for 'mix' command."""
    console.print("[yellow]Note:[/yellow] 'merge' is deprecated, use 'mix' instead.")
    mix(skills_arg, weights, output, description)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point for PSA CLI."""
    app()


if __name__ == "__main__":
    main()
