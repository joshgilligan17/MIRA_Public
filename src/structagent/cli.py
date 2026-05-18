"""Interactive CLI for MIRA."""

import os
import time
import importlib
from pathlib import Path
import click
import pyfiglet
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel

from structagent.agent import MiraAgent
from structagent.registry import get_registry
from structagent.prompts import EXAMPLE_QUERIES
from structagent.themes import THEMES, get_theme, set_theme, list_themes, theme_help_text, get_current_theme_name
from structagent import providers as providers_module

# Global console instance
_console = Console()
console = _console

# Provider presets for API configuration
PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_vars": ["OPENAI_API_KEY"],
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "env_vars": ["ANTHROPIC_API_KEY"],
        "default_model": "claude-3-5-haiku-20241022",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "env_vars": ["MINIMAX_API_KEY"],
        "default_model": "MiniMax-M2.7",
    },
    "azure": {
        "base_url": None,  # Must be explicitly provided
        "env_vars": ["AZURE_OPENAI_KEY"],
        "default_model": None,
    },
}

TOOL_MODULES = [
    "structagent.tools.structure_io",
    "structagent.tools.contacts",
    "structagent.tools.sasa",
    "structagent.tools.secondary_structure",
    "structagent.tools.interface",
    "structagent.tools.alignment",
    "structagent.tools.annotations",
    "structagent.tools.bfactor",
    "structagent.tools.charge",
    "structagent.tools.conservation",
    "structagent.tools.ramachandran",
    "structagent.tools.foldseek",
    "structagent.tools.dynamics",
    "structagent.tools.relaxation",
    "structagent.tools.interface_energy",
    "structagent.tools.pyrosetta_interface",
    "structagent.tools.renumber_pdb",
]


def initialize_tools():
    """Import tool modules so decorators register tools with the registry."""
    registry = get_registry()
    reload_modules = not registry.list_tools()
    for module_name in TOOL_MODULES:
        module = importlib.import_module(module_name)
        if reload_modules:
            importlib.reload(module)


def resolve_api_key(
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
) -> tuple[str, str, str]:
    """Resolve API configuration.

    Returns (resolved_api_key, resolved_base_url, resolved_model)

    Priority:
    1. Explicit api_key passed
    2. Env vars for specified provider
    3. Fallback to MiniMax defaults

    Raises:
        ValueError: If Azure provider is used without base_url
    """
    # If explicit api_key provided, use it
    if api_key:
        resolved_base_url = base_url or PROVIDER_PRESETS.get(provider, {}).get("base_url")
        resolved_model = model or PROVIDER_PRESETS.get(provider, {}).get("default_model")
        # Validate Azure requires base_url
        if provider == "azure" and not resolved_base_url:
            raise ValueError("Azure provider requires --base-url endpoint")
        return api_key, resolved_base_url, resolved_model

    # Check env vars for provider
    if provider and provider in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[provider]
        for env_var in preset["env_vars"]:
            key = os.environ.get(env_var)
            if key:
                resolved_base_url = base_url or preset["base_url"]
                # Validate Azure requires base_url
                if provider == "azure" and not resolved_base_url:
                    raise ValueError("Azure provider requires --base-url endpoint")
                return (key, resolved_base_url, model or preset["default_model"])

    # Fallback to MiniMax
    for env_var in ["MINIMAX_API_KEY", "OPENAI_API_KEY"]:
        key = os.environ.get(env_var)
        if key:
            return key, "https://api.minimax.io/v1", "MiniMax-M2.7"

    # No key found
    return None, None, None


def parse_toolsets(toolsets_str):
    """Parse comma-separated toolsets string into list."""
    if not toolsets_str:
        return None
    return [t.strip() for t in toolsets_str.split(",") if t.strip()]


def get_api_key(env_api_key=None, prompt_if_missing=True, provider=None):
    """Get API key from environment or prompt user.

    Checks in order:
    1. Passed env_api_key parameter
    2. Provider-specific env vars if provider is specified
    3. MINIMAX_API_KEY / OPENAI_API_KEY environment variables

    If none found and prompt_if_missing=True, prompts user securely.
    Returns None if no key found and not prompting.
    """
    # First check passed parameter
    if env_api_key:
        return env_api_key

    # Check provider-specific env vars
    if provider and provider in PROVIDER_PRESETS:
        for env_var in PROVIDER_PRESETS[provider]["env_vars"]:
            api_key = os.environ.get(env_var)
            if api_key:
                return api_key

    # Check environment variables (legacy fallback)
    for env_var in ("MINIMAX_API_KEY", "OPENAI_API_KEY"):
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key

    # Prompt user if allowed
    if prompt_if_missing:
        _console.print("\n[bold cyan]No API key found in environment.[/bold cyan]")
        _console.print("[dim]Your API key will be stored in memory only for this session.[/dim]")
        api_key = click.prompt(
            "Enter your API key",
            type=str,
            default="",
            show_default=False,
            hide_input=True,
        )
        if api_key.strip():
            return api_key.strip()

    return None


def print_banner(model, num_tools, theme=None):
    """Print startup banner with themed styling."""
    if theme is None:
        theme = get_theme()  # Uses themes.py's _current_theme
    ascii_banner = pyfiglet.figlet_format("MIRA", font="isometric1", width=120)
    byline = pyfiglet.figlet_format("Molecular Intelligence and Reasoning Agent", font="small", width=200)
    banner = f"""[bold {theme["banner"]}]{ascii_banner}[/bold {theme["banner"]}]
[bold {theme["secondary"]}]{byline}[/bold {theme["secondary"]}]
[{theme["dim"]}]┌─────────────────────────────────────────────────────────────┐[/{theme["dim"]}]
[{theme["dim"]}]│[/{theme["dim"]}] [bold {theme["primary"]}]Model:[/bold {theme["primary"]}] [{theme["secondary"]}]{model}[/{theme["secondary"]}]                               [{theme["dim"]}]│[/{theme["dim"]}]
[{theme["dim"]}]│[/{theme["dim"]}] [bold {theme["primary"]}]Tools:[/bold {theme["primary"]}]  [{theme["accent"]}]{num_tools}[/{theme["accent"]}] available                          [{theme["dim"]}]│[/{theme["dim"]}]
[{theme["dim"]}]│[/{theme["dim"]}] [dim]Type /help for commands, /theme to change colors           │[/dim]
[{theme["dim"]}]└─────────────────────────────────────────────────────────────┘[/{theme["dim"]}]"""
    _console.print(Panel(banner, expand=False, border_style=theme["primary"]))


def print_tools():
    """List available tools with descriptions."""
    registry = get_registry()
    schemas = registry.get_tool_schemas()
    theme = get_theme()  # Uses themes.py's _current_theme

    if not schemas:
        _console.print(f"[bold {theme['secondary']}]No tools registered.[/bold {theme['secondary']}]")
        return

    table = Table(
        title=f"[bold {theme['primary']}]Available Tools[/bold {theme['primary']}]",
        show_header=True,
        header_style=f"bold {theme['secondary']}",
        border_style=theme["primary"],
        row_styles=[theme["dim"], "#16213e"],
    )
    table.add_column(
        f"[bold {theme['secondary']}]Tool[/bold {theme['secondary']}]", style=theme["secondary"], no_wrap=True
    )
    table.add_column(f"[bold {theme['accent']}]Description[/bold {theme['accent']}]", style=theme["accent"])

    for schema in schemas:
        func = schema.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        table.add_row(name, desc)

    _console.print(table)


def print_examples():
    """Show example queries."""
    _console.print("\n[bold #ff00ff]Example Queries:[/bold #ff00ff]\n")

    for key, template in EXAMPLE_QUERIES.items():
        _console.print(f"  [cyan]{key}:[/cyan]")
        _console.print(f"    {template}")
        _console.print()


def save_trajectory(run, trajectory_dir):
    """Save an AgentRun to a JSONL file."""
    os.makedirs(trajectory_dir, exist_ok=True)
    import json

    timestamp = int(time.time() * 1000)
    filename = os.path.join(trajectory_dir, f"{timestamp}.jsonl")
    with open(filename, "w") as f:
        f.write(json.dumps(run.to_dict()) + "\n")
    return filename


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """MIRA — Molecular Intelligence and Reasoning Agent.

    Usage: mira [OPTIONS] COMMAND [ARGS]...

    By default, invokes the chat command if no subcommand is specified.
    Run `mira chat` to start interactive mode.
    """


@cli.command()
@click.option("--model", default=None, help="Model identifier (e.g., gpt-4o-mini, claude-3-5-haiku-20241022)")
@click.option("--base-url", default=None, help="API base URL")
@click.option("--api-key", default=None, help="API key (or set provider-specific env var)")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "minimax", "azure"]),
    default=None,
    help="API provider preset (openai, anthropic, minimax, azure)",
)
@click.option("--max-steps", default=15, help="Max tool-calling steps per query")
@click.option(
    "--toolsets",
    default=None,
    help="Comma-separated toolsets to enable: structure, analysis, dynamics (leave empty for all)",
)
@click.option("--save-trajectories", is_flag=True, help="Save runs to JSONL")
@click.option("--trajectory-dir", default="./trajectories", help="Directory for trajectory files")
@click.option("--verbose/--quiet", default=True)
@click.option("--plan-only", is_flag=True, help="Output plan without executing")
@click.option("--no-plan", is_flag=True, help="Skip planning, use direct ReAct")
@click.option(
    "--display",
    default="verbose",
    type=click.Choice(["normal", "verbose"]),
    help="Display mode: 'normal' for compact, 'verbose' for detailed",
)
@click.option("--timeout", default=120.0, type=float, help="Request timeout in seconds (default 120)")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature (default 0.0)")
def chat(
    model,
    base_url,
    api_key,
    provider,
    max_steps,
    toolsets,
    save_trajectories,
    trajectory_dir,
    verbose,
    plan_only,
    no_plan,
    display,
    timeout,
    temperature,
):
    """MIRA — Molecular Reasoning and Intelligence Agent."""
    toolsets_list = parse_toolsets(toolsets)

    # Validate --plan-only and --no-plan conflict
    if plan_only and no_plan:
        _console.print("[yellow]Warning: --plan-only and --no-plan conflict. Using --plan-only.[/yellow]")

    # Resolve API configuration
    try:
        resolved_api_key, resolved_base_url, resolved_model = resolve_api_key(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except ValueError as e:
        _console.print(f"[bold #ff3366]Error: {e}[/bold #ff3366]")
        return

    if not resolved_api_key:
        _console.print("[bold #ff3366]Error: No API key provided. Set provider env var or use --api-key[/bold #ff3366]")
        return

    # Use resolved values, falling back to explicit values or defaults
    final_model = resolved_model or model or "MiniMax-M2.7"
    final_base_url = resolved_base_url or base_url or "https://api.minimax.io/v1"

    # Initialize registry by importing tools
    initialize_tools()

    registry = get_registry()
    num_tools = len(registry.list_tools())

    # Print banner
    print_banner(final_model, num_tools)

    # Print tools
    _console.print()
    print_tools()

    # Initialize agent
    agent = MiraAgent(
        model=final_model,
        base_url=final_base_url,
        api_key=resolved_api_key,
        max_steps=max_steps,
        toolsets=toolsets_list,
        verbose=verbose,
        mode="react" if no_plan else "plan",
        display=display,
        timeout=timeout,
        temperature=temperature,
    )

    # Message history for multi-turn conversation
    message_history = []

    _console.print()

    while True:
        try:
            # Use click.prompt for better cross-platform compatibility
            user_input = click.prompt("\nYou", prompt_suffix=" > ", show_default=False)
        except (KeyboardInterrupt, EOFError):
            _console.print("\n[bold cyan]Exiting...[/bold cyan]")
            break

        user_input = user_input.strip()
        if not user_input:
            _console.print("[dim]Empty query. Type a question or /help for commands.[/dim]")
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                _console.print("[bold #00ff00]Goodbye![/bold #00ff00]")
                break
            elif cmd == "/tools":
                print_tools()
                continue
            elif cmd == "/help":
                _console.print(
                    """
[bold]Commands:[/bold]
  /tools     — list available tools with descriptions
  /theme     — show/change color theme (/theme <name>)
  /history   — show conversation summary
  /clear     — reset conversation history
  /save      — save current trajectory to JSONL
  /example   — show example queries
  /quit      — exit
"""
                )
                continue
            elif cmd == "/history":
                if not message_history:
                    _console.print("[dim]No conversation history.[/dim]")
                else:
                    _console.print(f"[dim]{len(message_history)} messages in history[/dim]")
                    for i, msg in enumerate(message_history):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            preview = content[:80] + "..." if len(content) > 80 else content
                        else:
                            preview = str(content)[:80]
                        _console.print(f"  [{i}] {role}: {preview}")
                continue
            elif cmd == "/clear":
                message_history = []
                _console.print("[dim]Conversation cleared.[/dim]")
                continue
            elif cmd == "/save":
                # Can't save without a run - prompt user to run a query first
                _console.print("[dim]Run a query first, then use /save to save the last run.[/dim]")
                continue
            elif cmd == "/example":
                print_examples()
                continue
            elif cmd.startswith("/theme"):
                # Handle /theme command
                parts = user_input.split()
                if len(parts) == 1:
                    # Show available themes
                    _console.print(theme_help_text())
                elif len(parts) == 2:
                    # Try to set theme
                    theme_name = parts[1].lower()
                    if set_theme(theme_name):
                        theme = get_theme(theme_name)
                        _console.print(f"[{theme['success']}]Theme changed to {theme['name']}[/{theme['success']}]")
                        # Re-print banner with new theme
                        print_banner(final_model, num_tools)
                    else:
                        _console.print(f"[{get_theme()['error']}]Unknown theme: {theme_name}[/{get_theme()['error']}]")
                        _console.print("Type /theme to see available themes.")
                else:
                    _console.print("[error]Usage: /theme [name][/error]")
                continue
            else:
                _console.print(f"[bold cyan]Unknown command: {user_input}[/bold cyan]")
                _console.print("Type /help for available commands.")
                continue

        # Regular query - run the agent
        try:
            # Handle --plan-only mode
            if plan_only:
                plan = agent.create_plan(user_input)
                plan_text = f"[bold cyan]Analysis Plan ({len(plan.get('steps', []))} steps):[/bold cyan]\n\n"
                if plan.get("reasoning"):
                    plan_text += f"[dim]{plan['reasoning']}[/dim]\n\n"
                if plan.get("steps"):
                    plan_text += "\n".join(
                        f"  {i + 1}. {s['tool']}({s.get('args', {})}) — {s.get('purpose', '')}"
                        for i, s in enumerate(plan["steps"])
                    )
                _console.print(Panel(plan_text, title="Plan"))
                continue

            # Show plan in verbose mode before execution (unless --no-plan)
            if verbose and not no_plan:
                plan = agent.create_plan(user_input)
                _console.print(
                    Panel(
                        f"[bold cyan]Analysis Plan ({len(plan.get('steps', []))} steps):[/bold cyan]\n\n"
                        + (f"[dim]{plan.get('reasoning', '')}[/dim]\n\n" if plan.get("reasoning") else "")
                        + "\n".join(
                            f"  {i + 1}. {s['tool']}({s.get('args', {})}) — {s.get('purpose', '')}"
                            for i, s in enumerate(plan.get("steps", []))
                        ),
                        title="Plan",
                    )
                )
                _console.print("Executing...")

            start_time = time.time()
            run = agent.chat(
                query=user_input,
                message_history=message_history,
            )
            wall_time = time.time() - start_time

            # Add user message and assistant responses to history
            message_history.append({"role": "user", "content": user_input})
            message_history.append({"role": "assistant", "content": run.final_answer})

            # Print final answer with markdown rendering
            _console.print()
            if run.final_answer:
                md = Markdown(run.final_answer)
                _console.print(md)

            # Print status
            status = f"[dim]{run.total_steps} steps"
            if run.total_input_tokens or run.total_output_tokens:
                total_tokens = run.total_input_tokens + run.total_output_tokens
                status += f" │ {total_tokens:,} tokens"
            status += f" │ {wall_time:.1f}s[/dim]"
            _console.print(f"\n{status}")

            # Auto-save if enabled
            if save_trajectories:
                filename = save_trajectory(run, trajectory_dir)
                _console.print(f"[dim]Trajectory saved to {filename}[/dim]")

        except KeyboardInterrupt:
            _console.print("\n[bold cyan]Interrupted. Returning to prompt.[/bold cyan]")
            continue
        except Exception as e:
            from rich.markup import escape

            error_msg = f"Error: {str(e)}"
            _console.print(f"[bold #ff3366]{escape(error_msg)}[/bold #ff3366]")
            continue

    _console.print("[dim]Session ended.[/dim]")


@cli.command()
@click.argument("query")
@click.option("--folder", type=click.Path(exists=True), help="Folder containing PDB files")
@click.option("--pdb-ids", type=str, help="Comma-separated PDB IDs (e.g., '1UBQ,2HHI')")
@click.option("--glob", default="*", help="Glob pattern for structure files (default: *)")
@click.option(
    "--rank-by",
    default="stability",
    type=click.Choice(
        [
            "stability",
            "buried_surface_area",
            "n_interface_residues",
            "mean_bfactor",
            "std_bfactor",
            "n_buried",
            "n_exposed",
            "interface_energy",
        ]
    ),
    help="Ranking criterion",
)
@click.option("--parallel/--sequential", default=True, help="Process structures in parallel (default)")
@click.option("--max-workers", type=int, default=4, help="Max parallel workers (default 4)")
@click.option("--max-steps", default=15, help="Max tool-calling steps per structure")
@click.option("--save-trajectories", is_flag=True, help="Save runs to JSONL")
@click.option("--trajectory-dir", default="./trajectories", help="Directory for trajectory files")
@click.option("--verbose/--quiet", default=False)
@click.option("--model", default=None, help="Model identifier (overrides provider default)")
@click.option("--base-url", default=None, help="API base URL (overrides provider default)")
@click.option("--api-key", default=None, help="API key (overrides provider env var)")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "minimax", "azure"]),
    default=None,
    help="API provider preset (openai, anthropic, minimax, azure)",
)
@click.option("--timeout", default=120.0, type=float, help="Request timeout in seconds (default 120)")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature (default 0.0)")
@click.option(
    "--subagent/--no-subagent", default=False, help="Use new subagent-based execution (recommended for large batches)"
)
def batch(
    query,
    folder,
    pdb_ids,
    glob,
    rank_by,
    parallel,
    max_workers,
    max_steps,
    save_trajectories,
    trajectory_dir,
    verbose,
    model,
    base_url,
    api_key,
    provider,
    timeout,
    temperature,
    subagent,
):
    """Batch analysis mode: analyze multiple PDB structures with joint ranking.

    Examples:

        mira batch --folder ./binders "analyze interface stability"

        mira batch --pdb-ids "1BRS,1BRC,1BRD" "analyze stability"

        mira batch --folder ./antibodies --glob "*.pdb" --rank-by interface_energy "rank by binding energy"

        mira batch --folder ./pdbs --provider openai --model gpt-4o-mini "Analyze"

    Use --subagent for the new parallel subagent architecture (recommended for
    large batches with many structures).
    """
    # Resolve API configuration
    try:
        resolved_api_key, resolved_base_url, resolved_model = resolve_api_key(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except ValueError as e:
        _console.print(f"[bold #ff3366]Error: {e}[/bold #ff3366]")
        return

    if not resolved_api_key:
        _console.print("[bold #ff3366]Error: No API key provided. Set provider env var or use --api-key[/bold #ff3366]")
        return

    final_model = resolved_model or "MiniMax-M2.7"
    final_base_url = resolved_base_url or "https://api.minimax.io/v1"

    # Initialize tools
    initialize_tools()

    from structagent.batch import BatchRunner, BatchSynthesisEngine
    from structagent.agent import MiraAgent

    # Parse PDB IDs
    pdb_id_list = []
    pdb_path_list = []

    if pdb_ids:
        pdb_id_list = [p.strip().upper() for p in pdb_ids.split(",") if p.strip()]
        pdb_path_list = [None] * len(pdb_id_list)
    elif folder:
        # Discover PDBs from folder
        agent = MiraAgent(
            model=final_model,
            base_url=final_base_url,
            api_key=resolved_api_key,
            max_steps=max_steps,
            verbose=verbose,
            mode="plan",
            timeout=timeout,
            temperature=temperature,
        )
        runner = BatchRunner(agent, max_workers=max_workers)
        discovered = runner.discover_pdbs(folder, glob)
        if not discovered:
            _console.print("[bold red]No PDB files found in folder.[/bold red]")
            return
        pdb_id_list, pdb_path_list = zip(*discovered)
    else:
        _console.print("[bold red]Error: Must specify --folder or --pdb-ids[/bold red]")
        return

    if not pdb_id_list:
        _console.print("[bold red]Error: No structures to analyze.[/bold red]")
        return

    # Initialize agent if not already done
    if folder:
        # Agent already created above for PDB discovery
        pass
    else:
        agent = MiraAgent(
            model=final_model,
            base_url=final_base_url,
            api_key=resolved_api_key,
            max_steps=max_steps,
            verbose=verbose,
            mode="plan",
            timeout=timeout,
            temperature=temperature,
        )

    # Show banner
    _console.print(f"[bold #ff00ff]MIRA Batch Analysis[/bold #ff00ff]")
    execution_mode = "subagent" if subagent else "legacy"
    _console.print(f"[dim]Analyzing {len(pdb_id_list)} structures... ({execution_mode} mode)[/dim]")
    _console.print(f"[dim]Ranking by: {rank_by}[/dim]\n")

    # Create batch runner
    if subagent:
        # Use new subagent-based execution via OrchestratorAgent
        runner = BatchRunner(
            agent=None,  # OrchestratorAgent creates its own subagents
            max_workers=max_workers if parallel else 1,
            max_subagents=max_workers if parallel else 1,
            model=final_model,
            use_subagents=True,
            base_url=final_base_url,
            api_key=resolved_api_key,
            max_steps=max_steps,
            timeout=timeout,
            temperature=temperature,
            verbose=verbose,
        )
    else:
        # Use legacy single-plan approach with MiraAgent
        runner = BatchRunner(agent, max_workers=max_workers if parallel else 1)

    # Run batch analysis
    try:
        batch_result = runner.run(
            query=query, pdb_ids=list(pdb_id_list), pdb_paths=list(pdb_path_list), rank_by=rank_by
        )

        # Generate synthesis
        if batch_result.structure_results:
            _console.print(f"\n[bold cyan]Generating comparative analysis...[/bold cyan]")
            # Create provider for synthesis using resolved config
            synthesis_provider = providers_module.create_provider(
                provider_name=provider or "openai",
                api_key=resolved_api_key,
                base_url=final_base_url,
                timeout=timeout,
                temperature=temperature,
            )
            synthesis_engine = BatchSynthesisEngine()
            synthesis = synthesis_engine.synthesize(
                batch_result,
                synthesis_provider,
                model=final_model,
                temperature=temperature,
            )
            batch_result.synthesis = synthesis

        # Display results
        _console.print(
            Panel(
                f"[bold]Ranking by {batch_result.ranking_criterion}:[/bold]\n"
                + "\n".join(
                    f"  {i + 1}. [cyan]{pid}[/cyan]: {score:.2f}" for i, (pid, score) in enumerate(batch_result.ranking)
                ),
                title="Results",
            )
        )

        # Display synthesis
        if batch_result.synthesis:
            _console.print("\n")
            _console.print(Markdown(batch_result.synthesis))

        # Display stats
        _console.print(
            f"\n[dim]{len(batch_result.structure_results)} structures analyzed | "
            f"{batch_result.total_wall_time:.1f}s | "
            f"{batch_result.total_tokens:,} tokens[/dim]"
        )

        # Save trajectories
        if save_trajectories:
            save_batch_trajectory(batch_result, trajectory_dir)

    except Exception as e:
        import traceback

        error_msg = str(e).replace("[", "[[]").replace("]", "[]]")
        _console.print(f"[bold #ff0066]Error during batch analysis:[/bold #ff0066] {error_msg}")
        _console.print(f"[dim]{traceback.format_exc()}[/dim]")


def save_batch_trajectory(batch_result, trajectory_dir):
    """Save batch result to JSONL file."""
    import json
    from pathlib import Path

    Path(trajectory_dir).mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = Path(trajectory_dir) / f"batch_{timestamp}.jsonl"

    data = {
        "query": batch_result.query,
        "ranking_criterion": batch_result.ranking_criterion,
        "ranking": batch_result.ranking,
        "synthesis": batch_result.synthesis,
        "total_wall_time": batch_result.total_wall_time,
        "total_tokens": batch_result.total_tokens,
        "structure_results": [
            {
                "pdb_id": sr.pdb_id,
                "pdb_path": sr.pdb_path,
                "success": sr.success,
                "error": sr.error,
                "metrics": sr.metrics,
                "final_answer": sr.run.final_answer if sr.run else "",
            }
            for sr in batch_result.structure_results
        ],
    }

    with open(filename, "w") as f:
        f.write(json.dumps(data) + "\n")

    _console.print(f"[dim]Batch trajectory saved to {filename}[/dim]")


@cli.command("batch-analyze", hidden=True)
@click.argument("target", type=str)
@click.argument("binders", type=click.Path(exists=True))
@click.option(
    "--query",
    "-q",
    type=str,
    default="Analyze the interface between target and binder structures. Compute interface metrics including buried surface area, interface residues, and interface energy.",
    help="Analysis query",
)
@click.option("--max-subagents", type=int, default=4, help="Max parallel subagents (default 4)")
@click.option(
    "--rank-by",
    default="interface_energy",
    type=click.Choice(
        [
            "stability",
            "buried_surface_area",
            "n_interface_residues",
            "mean_bfactor",
            "std_bfactor",
            "n_buried",
            "n_exposed",
            "interface_energy",
        ]
    ),
    help="Ranking criterion",
)
@click.option("--model", default=None, help="Model identifier (overrides provider default)")
@click.option("--base-url", default=None, help="API base URL (overrides provider default)")
@click.option("--api-key", default=None, help="API key (overrides provider env var)")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "minimax", "azure"]),
    default=None,
    help="API provider preset (openai, anthropic, minimax, azure)",
)
@click.option("--max-steps", default=15, help="Max tool-calling steps per structure")
@click.option("--timeout", default=120.0, type=float, help="Request timeout in seconds (default 120)")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature (default 0.0)")
@click.option("--save-trajectories", is_flag=True, help="Save runs to JSONL")
@click.option("--trajectory-dir", default="./trajectories", help="Directory for trajectory files")
@click.option("--output", "-o", type=click.Path(), default=None, help="Save results to markdown file")
@click.option("--verbose/--quiet", default=False)
def batch_analyze(
    target,
    binders,
    query,
    max_subagents,
    rank_by,
    model,
    base_url,
    api_key,
    provider,
    max_steps,
    timeout,
    temperature,
    save_trajectories,
    trajectory_dir,
    output,
    verbose,
):
    """Advanced batch analysis with subagent support.

    TARGET: Target structure (PDB ID or path to local PDB file)
    BINDERS: Folder containing binder PDB files

    This command uses the OrchestratorAgent to spawn multiple subagents
    for parallel analysis of target-binder pairs.

    Examples:

        mira batch-analyze 1AH9 ./binders

        mira batch-analyze 1AH9 ./binders -q "Analyze interface stability"

        mira batch-analyze 1AH9 ./binders --max-subagents 8 --rank-by buried_surface_area
    """
    # Resolve API configuration
    try:
        resolved_api_key, resolved_base_url, resolved_model = resolve_api_key(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except ValueError as e:
        _console.print(f"[bold #ff3366]Error: {e}[/bold #ff3366]")
        return

    if not resolved_api_key:
        _console.print("[bold #ff3366]Error: No API key provided. Set provider env var or use --api-key[/bold #ff3366]")
        return

    final_model = resolved_model or "MiniMax-M2.7"
    final_base_url = resolved_base_url or "https://api.minimax.io/v1"

    initialize_tools()

    from structagent.orchestrator import OrchestratorAgent
    from structagent.batch import BatchRunner, BatchSynthesisEngine, ResultAggregator, StructureResult, BatchResult
    from structagent.agent import AgentRun

    # Determine if target is a local file or PDB ID
    target_path = None
    target_id = target
    if os.path.isfile(target):
        target_path = os.path.abspath(target)
        target_id = os.path.splitext(os.path.basename(target))[0].upper()

    # Discover binder PDBs
    binder_path = Path(binders)
    binder_pdbs = []
    for f in binder_path.glob("*.pdb"):
        pdb_id = f.stem.upper()
        binder_pdbs.append((pdb_id, str(f.absolute())))
    binder_pdbs.sort(key=lambda x: x[0])

    if not binder_pdbs:
        _console.print(f"[bold red]No PDB files found in {binders}[/bold red]")
        return

    # Build structure list: target + each binder as separate analysis
    # For each binder, we analyze the target-binder pair
    pdb_ids = [target_id] * len(binder_pdbs)
    pdb_paths = [target_path] * len(binder_pdbs)
    # Append binder info to query for each analysis
    # Actually, for batch-analyze we run target vs each binder

    # For a proper target-binder analysis, we need to analyze each pair
    # Let's set up structures where each structure is a target-binder pair
    structure_list = []
    for binder_id, binder_path in binder_pdbs:
        # Each entry is (pdb_id, pdb_path)
        # We analyze target-binder as a complex
        structure_list.append((f"{target_id}_{binder_id}", f"{target_path},{binder_path}"))

    # Flatten for the orchestrator
    all_pdb_ids = [s[0] for s in structure_list]
    all_pdb_paths = [s[1] for s in structure_list]

    # Show banner
    _console.print(f"[bold #ff00ff]MIRA Batch Analyze (Subagent Mode)[/bold #ff00ff]")
    _console.print(f"[dim]Target: {target_id} | {len(binder_pdbs)} binders[/dim]")
    _console.print(f"[dim]Max subagents: {max_subagents} | Ranking by: {rank_by}[/dim]")
    _console.print()

    # Create orchestrator
    orchestrator = OrchestratorAgent(
        max_subagents=max_subagents,
        model=final_model,
        base_url=final_base_url,
        api_key=resolved_api_key,
        max_steps=max_steps,
        timeout=timeout,
        temperature=temperature,
        verbose=verbose,
    )

    # Run analysis
    start_time = time.time()
    try:
        _console.print(f"[dim]Starting parallel analysis with {max_subagents} subagents...[/dim]")

        # Run with orchestrator
        subagent_results = orchestrator.run_synchronous(query, all_pdb_ids, all_pdb_paths)

        # Convert SubagentResult to StructureResult for compatibility
        aggregator = ResultAggregator(rank_by)
        results: list[StructureResult] = []
        total_tokens = 0

        for sar in subagent_results:
            if sar.success:
                run = AgentRun(
                    query=query,
                    steps=sar.steps,
                    final_answer=sar.final_answer,
                    total_steps=len(sar.steps),
                    total_input_tokens=sar.total_input_tokens,
                    total_output_tokens=sar.total_output_tokens,
                    wall_time_seconds=0.0,
                    model=final_model,
                )
                total_tokens += sar.total_input_tokens + sar.total_output_tokens
            else:
                run = None

            structure_result = StructureResult(
                pdb_id=sar.pdb_id,
                pdb_path=sar.pdb_path,
                run=run,
                metrics=sar.metrics,
                success=sar.success,
                error=sar.error,
            )
            results.append(structure_result)
            aggregator.add_result(structure_result)

        # Sort results to match ranking order
        ranking_dict = dict(aggregator.get_ranking())
        results.sort(key=lambda r: ranking_dict.get(r.pdb_id, float("inf")))

        total_time = time.time() - start_time

        # Create BatchResult for synthesis
        batch_result = BatchResult(
            query=query,
            structure_results=results,
            ranking=aggregator.get_ranking(),
            ranking_criterion=rank_by,
            synthesis="",
            total_wall_time=total_time,
            total_tokens=total_tokens,
            orchestrator_used=True,
            subagent_results=subagent_results,
        )

        # Generate synthesis
        if results:
            _console.print(f"\n[bold cyan]Generating comparative analysis...[/bold cyan]")
            synthesis_provider = providers_module.create_provider(
                provider_name=provider or "openai",
                api_key=resolved_api_key,
                base_url=final_base_url,
                timeout=timeout,
                temperature=temperature,
            )
            synthesis_engine = BatchSynthesisEngine()
            synthesis = synthesis_engine.synthesize(
                batch_result,
                synthesis_provider,
                model=final_model,
                temperature=temperature,
            )
            batch_result.synthesis = synthesis

        # Display results table
        _console.print()
        table = Table(
            title=f"[bold #ff00ff]Ranking by {rank_by}[/bold #ff00ff]",
            show_header=True,
            header_style="bold #ff00ff",
            border_style="#ff00ff",
        )
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Structure", style="cyan")
        table.add_column("Score", style="green", justify="right")

        for i, (pdb_id, score) in enumerate(batch_result.ranking, 1):
            # Parse target_binder format
            display_id = pdb_id.replace("_", " + ")
            table.add_row(str(i), display_id, f"{score:.2f}")

        _console.print(table)

        # Display synthesis
        if batch_result.synthesis:
            _console.print()
            _console.print(Markdown(batch_result.synthesis))

        # Display stats
        _console.print(
            f"\n[dim]{len(results)} structures analyzed | "
            f"{batch_result.total_wall_time:.1f}s | "
            f"{batch_result.total_tokens:,} tokens[/dim]"
        )

        # Save trajectories
        if save_trajectories:
            save_batch_trajectory(batch_result, trajectory_dir)

        # Save results to markdown file
        if output:
            with open(output, "w") as f:
                f.write(f"# Batch Analysis Results\n\n")
                f.write(f"**Query:** {query}\n\n")
                f.write(f"**Target:** {target_id}\n\n")
                f.write(f"**Binders:** {len(binder_pdbs)}\n\n")
                f.write(f"**Ranking Criterion:** {rank_by}\n\n")
                f.write(f"**Total Time:** {batch_result.total_wall_time:.1f}s\n\n")
                f.write(f"**Total Tokens:** {batch_result.total_tokens:,}\n\n")
                f.write(f"---\n\n")
                f.write(f"## Rankings\n\n")
                f.write(f"| Rank | Structure | Score |\n")
                f.write(f"|------|-----------|-------|\n")
                for i, (pdb_id, score) in enumerate(batch_result.ranking, 1):
                    display_id = pdb_id.replace("_", " + ")
                    f.write(f"| {i} | {display_id} | {score:.2f} |\n")
                f.write(f"\n---\n\n")
                if batch_result.synthesis:
                    f.write(f"## Comparative Analysis\n\n")
                    f.write(f"{batch_result.synthesis}\n\n")
                f.write(f"---\n\n")
                f.write(f"## Detailed Results\n\n")
                for sr in results:
                    f.write(f"### {sr.pdb_id}\n\n")
                    if sr.success:
                        f.write(f"**Status:** Success\n\n")
                        if sr.metrics:
                            f.write(f"**Metrics:**\n\n")
                            for k, v in sr.metrics.items():
                                f.write(f"- {k}: {v}\n")
                            f.write(f"\n")
                        if sr.run and sr.run.final_answer:
                            f.write(f"**Analysis:** {sr.run.final_answer[:500]}")
                            if len(sr.run.final_answer) > 500:
                                f.write(f"... [truncated]")
                            f.write(f"\n\n")
                    else:
                        f.write(f"**Status:** Failed\n\n")
                        if sr.error:
                            f.write(f"**Error:** {sr.error}\n\n")
                f.write(f"\n*Generated by MIRA*\n")
            _console.print(f"\n[green]Results saved to {output}[/green]")

    except Exception as e:
        import traceback

        error_msg = str(e).replace("[", "[[]").replace("]", "[]]")
        _console.print(f"[bold #ff0066]Error during batch analysis:[/bold #ff0066] {error_msg}")
        _console.print(f"[dim]{traceback.format_exc()}[/dim]")


@cli.command(hidden=True)
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--model", default=None, help="Model identifier")
@click.option("--base-url", default=None, help="API base URL")
@click.option("--api-key", default=None, help="API key")
@click.option("--provider", type=click.Choice(["openai", "anthropic", "minimax", "azure"]), default=None)
@click.option("--max-steps", default=15)
@click.option("--toolsets", default=None, help="Comma-separated toolsets")
def web(host, port, model, base_url, api_key, provider, max_steps, toolsets):
    """Launch web GUI with chat + PDB viewer.

    Opens http://localhost:8000 in your browser.
    """
    import uvicorn
    from structagent.web.server import app

    initialize_tools()

    # Store config in app state for sessions to use
    app.state.agent_config = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "provider": provider,
        "max_steps": max_steps,
        "toolsets": parse_toolsets(toolsets),
    }

    uvicorn.run(app, host=host, port=port)


@cli.command("binder-design", hidden=True)
@click.argument("target", type=str)
@click.argument("candidates", type=click.Path(exists=True))
@click.option(
    "--strategy",
    "-s",
    type=str,
    required=True,
    help="Design strategy for targeting the binder (e.g., 'targeting the epitope around residues 50-60')",
)
@click.option(
    "--stage",
    type=click.Choice(["1", "2", "both"]),
    default="both",
    help="Which stage(s) to run: 1=target analysis, 2=candidate analysis, both (default)",
)
@click.option(
    "--target-analysis-output", type=click.Path(), default=None, help="Save Stage 1 target analysis to JSON file"
)
@click.option(
    "--target-analysis-input",
    type=click.Path(),
    default=None,
    help="Load Stage 1 target analysis from JSON file (for stage 2 only)",
)
@click.option("--max-subagents", type=int, default=4, help="Max parallel subagents (default 4)")
@click.option(
    "--rank-by",
    default="interface_energy",
    type=click.Choice(
        [
            "stability",
            "buried_surface_area",
            "n_interface_residues",
            "mean_bfactor",
            "std_bfactor",
            "n_buried",
            "n_exposed",
            "interface_energy",
            "shape_complementarity",
        ]
    ),
    help="Ranking criterion for candidates",
)
@click.option("--model", default=None, help="Model identifier (overrides provider default)")
@click.option("--base-url", default=None, help="API base URL (overrides provider default)")
@click.option("--api-key", default=None, help="API key (overrides provider env var)")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "minimax", "azure"]),
    default=None,
    help="API provider preset (openai, anthropic, minimax, azure)",
)
@click.option("--max-steps", default=15, help="Max tool-calling steps per structure")
@click.option("--timeout", default=120.0, type=float, help="Request timeout in seconds (default 120)")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature (default 0.0)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Save results to markdown file")
@click.option("--verbose/--quiet", default=False)
def binder_design(
    target,
    candidates,
    strategy,
    stage,
    target_analysis_output,
    target_analysis_input,
    max_subagents,
    rank_by,
    model,
    base_url,
    api_key,
    provider,
    max_steps,
    timeout,
    temperature,
    output,
    verbose,
):
    """Two-stage binder design analysis.

    TARGET: Target structure (PDB ID or path to local PDB file) - typically the receptor.
    CANDIDATES: Folder containing candidate binder PDB files to analyze.

    This command runs a two-stage analysis:

    Stage 1 (Target Analysis):
        Analyzes the target structure to identify:
        - Hotspot regions (high-contact interface residues)
        - Flexible regions (high B-factors, hinge regions)
        - Surface-exposed regions
        - Structural quality (Ramachandran)

    Stage 2 (Informed Candidate Analysis):
        Uses Stage 1 results to inform analysis of candidate binders,
        focusing on complementarity to target's identified features.

    Examples:

        mira binder-design 5ELI ./test_binders --strategy "target hot spot at interface"

        mira binder-design 5ELI ./test_binders --strategy "targeting residues 50-60" --stage 1

        mira binder-design 5ELI ./test_binders --stage 2 --target-analysis-input target_analysis.json

        mira binder-design 5ELI ./test_binders --stage both --max-subagents 8 --output results.md
    """
    # Resolve API configuration
    try:
        resolved_api_key, resolved_base_url, resolved_model = resolve_api_key(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except ValueError as e:
        _console.print(f"[bold #ff3366]Error: {e}[/bold #ff3366]")
        return

    if not resolved_api_key:
        _console.print("[bold #ff3366]Error: No API key provided. Set provider env var or use --api-key[/bold #ff3366]")
        return

    final_model = resolved_model or "MiniMax-M2.7"
    final_base_url = resolved_base_url or "https://api.minimax.io/v1"

    initialize_tools()

    from structagent.binder_design import TargetAnalyzer, InformedBatchRunner, TargetAnalysisReport
    from structagent.batch import BatchRunner, ResultAggregator, StructureResult, BatchResult
    from structagent.synthesis import BatchSynthesisEngine
    from structagent.agent import AgentRun

    # Determine if target is a local file or PDB ID
    target_path = None
    target_id = target
    if os.path.isfile(target):
        target_path = os.path.abspath(target)
        target_id = os.path.splitext(os.path.basename(target))[0].upper()

    # Discover candidate PDBs
    candidate_path = Path(candidates)
    candidate_pdbs = []
    for f in candidate_path.glob("*.pdb"):
        pdb_id = f.stem.upper()
        candidate_pdbs.append((pdb_id, str(f.absolute())))
    candidate_pdbs.sort(key=lambda x: x[0])

    if not candidate_pdbs:
        _console.print(f"[bold red]No PDB files found in {candidates}[/bold red]")
        return

    # Show banner
    _console.print(f"[bold #ff00ff]MIRA Binder Design (Two-Stage Mode)[/bold #ff00ff]")
    _console.print(f"[dim]Target: {target_id} | {len(candidate_pdbs)} candidates[/dim]")
    _console.print(f"[dim]Stage: {stage} | Strategy: {strategy}[/dim]")
    _console.print()

    # Stage 1: Target Analysis
    target_report = None

    if stage in ("1", "both"):
        _console.print(f"[bold cyan]Stage 1: Analyzing target {target_id}...[/bold cyan]")

        analyzer = TargetAnalyzer(
            model=final_model,
            base_url=final_base_url,
            api_key=resolved_api_key,
            timeout=timeout,
        )

        try:
            target_report = analyzer.analyze(
                target_id=target_id,
                target_path=target_path,
                design_strategy=strategy,
            )

            _console.print(f"[green]Target analysis complete.[/green]")
            _console.print(f"[dim]  - Hotspots: {len(target_report.hotspots)}[/dim]")
            _console.print(f"[dim]  - Flexible regions: {len(target_report.flexible_regions)}[/dim]")
            _console.print(f"[dim]  - Surface regions: {len(target_report.surface_regions)}[/dim]")
            _console.print(f"[dim]  - Focus: {', '.join(target_report.recommended_analysis_focus) or 'general'}[/dim]")

            # Save if requested
            if target_analysis_output:
                target_report.save(target_analysis_output)
                _console.print(f"[dim]Saved target analysis to {target_analysis_output}[/dim]")

            if stage == "1":
                _console.print("\n[bold cyan]Stage 1 complete. Run with --stage 2 to analyze candidates.[/bold cyan]")
                return

        except Exception as e:
            import traceback

            error_msg = str(e).replace("[", "[[]").replace("]", "[]]")
            _console.print(f"[bold #ff0066]Error during target analysis:[/bold #ff0066] {error_msg}")
            _console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return

    # Load target report if in stage 2 only mode
    if stage == "2":
        if target_analysis_input:
            try:
                target_report = TargetAnalysisReport.load(target_analysis_input)
                _console.print(f"[dim]Loaded target analysis from {target_analysis_input}[/dim]")
            except Exception as e:
                _console.print(f"[bold #ff0066]Error loading target analysis: {e}[/bold #ff0066]")
                return
        else:
            _console.print("[bold #ff0066]Error: --target-analysis-input required for stage 2[/bold #ff0066]")
            return

    # Stage 2: Informed Batch Analysis
    if stage in ("2", "both") and target_report:
        _console.print(f"\n[bold cyan]Stage 2: Analyzing {len(candidate_pdbs)} candidates...[/bold cyan]")

        runner = InformedBatchRunner(
            target_report=target_report,
            model=final_model,
            base_url=final_base_url,
            api_key=resolved_api_key,
            timeout=timeout,
            max_subagents=max_subagents,
        )

        try:
            batch_result = runner.run(
                candidate_folder=str(candidates),
                rank_by=rank_by,
            )

            # Generate target-informed synthesis
            _console.print(f"[dim]Generating synthesis...[/dim]")
            synthesis_engine = BatchSynthesisEngine(
                model=final_model,
                api_key=resolved_api_key,
                base_url=final_base_url,
            )
            synthesis_provider = providers_module.create_provider(
                provider_name=provider or "openai",
                api_key=resolved_api_key,
                base_url=final_base_url,
                timeout=timeout,
                temperature=temperature,
            )
            synthesis = synthesis_engine.synthesize_with_target(
                batch_result=batch_result,
                provider=synthesis_provider,
                model=final_model,
                temperature=temperature,
            )
            batch_result.synthesis = synthesis

            _console.print(f"[green]Batch analysis complete.[/green]")

            # Print ranking summary
            _console.print(f"\n[bold]Ranking (by {rank_by}):[/bold]")
            for i, (pdb_id, score) in enumerate(batch_result.ranking[:5], 1):
                _console.print(f"  [dim]{i}.[/dim] {pdb_id}: {score:.2f}")

            # Print synthesis
            if synthesis:
                _console.print("\n[bold]Synthesis:[/bold]")
                _console.print()
                from rich.markdown import Markdown

                md = Markdown(synthesis)
                _console.print(md)

            # Save if requested
            if output:
                with open(output, "w") as f:
                    f.write(f"# Binder Design Analysis: {target_id}\n\n")
                    f.write(f"## Target: {target_id}\n")
                    f.write(f"## Strategy: {strategy}\n\n")
                    f.write(f"## Target Analysis Summary\n{target_report.summary}\n\n")
                    f.write(f"## Candidate Rankings\n")
                    for i, (pdb_id, score) in enumerate(batch_result.ranking, 1):
                        f.write(f"{i}. {pdb_id}: {score:.2f}\n")
                    f.write(f"\n## Synthesis\n{synthesis}\n")
                _console.print(f"\n[dim]Results saved to {output}[/dim]")

        except Exception as e:
            import traceback

            error_msg = str(e).replace("[", "[[]").replace("]", "[]]")
            _console.print(f"[bold #ff0066]Error during batch analysis:[/bold #ff0066] {error_msg}")
            _console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    cli()
