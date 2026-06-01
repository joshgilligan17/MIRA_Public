"""Execution adapters for real design-model backends.

The adapters do not simulate generation. They either run an installed model
entrypoint or return a configuration-required result that tells the operator
which real backend is missing.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_DESIGN_SUFFIXES = {".pdb", ".cif", ".mmcif"}
SEQUENCE_SUFFIXES = {".fa", ".fasta"}


@dataclass
class DesignRequest:
    library: str
    target_path: Path | None
    output_dir: Path
    project_id: str
    run_id: str
    prompt: str
    chain_id: str = ""
    num_designs: int = 5
    seed: int = 0
    temperature: str = "0.1"
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedDesign:
    status: str
    command: str | None
    parameters: dict[str, Any]
    error: str | None = None


@dataclass
class DesignExecution:
    success: bool
    status: str
    generated_structure_paths: list[Path] = field(default_factory=list)
    generated_sequences: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    logs: str = ""
    error: str | None = None


def prepare_design(request: DesignRequest) -> PreparedDesign:
    """Prepare a concrete execution plan for a design backend."""

    library = request.library.lower().strip()
    if library == "proteinmpnn":
        return _prepare_proteinmpnn(request)
    if library == "ligandmpnn":
        return _prepare_ligandmpnn(request)
    if library == "foldingdiff":
        return _prepare_foldingdiff(request)
    if library == "rfdiffusion":
        return _prepare_rfdiffusion(request)
    if library == "bindcraft":
        return _prepare_bindcraft(request)
    return _prepare_custom(request)


def execute_design(parameters: dict[str, Any], output_dir: Path) -> DesignExecution:
    """Run a prepared design command and collect generated artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timeout = int(os.getenv("MIRA_DESIGN_TIMEOUT_SECONDS", "3600"))
    attempt_logs: list[str] = []
    last_error = ""
    for index, attempt in enumerate(_execution_attempts(parameters), start=1):
        label = str(attempt.get("label") or f"attempt_{index}")
        try:
            completed = _run_design_attempt(attempt, output_dir, timeout)
        except ValueError as exc:
            return DesignExecution(success=False, status="configuration_required", error=str(exc))
        except Exception as exc:
            last_error = str(exc)
            attempt_logs.append(f"[{label}] {last_error}")
            continue

        command = _attempt_command(attempt)
        logs = _tail_text("\n".join(part for part in [completed.stdout, completed.stderr] if part), 12000)
        attempt_logs.append(f"[{label}] {command}\n{logs}".strip())
        if completed.returncode != 0:
            last_error = (completed.stderr or completed.stdout or f"Exited with {completed.returncode}")[-4000:]
            continue

        structure_paths = [
            path
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_DESIGN_SUFFIXES
        ]
        return DesignExecution(
            success=True,
            status="completed",
            generated_structure_paths=structure_paths,
            generated_sequences=_collect_sequences(output_dir),
            artifacts=_collect_artifacts(output_dir),
            logs=_tail_text("\n\n".join(attempt_logs), 12000),
        )

    return DesignExecution(
        success=False,
        status="failed",
        artifacts=_collect_artifacts(output_dir),
        logs=_tail_text("\n\n".join(attempt_logs), 12000),
        error=(last_error or "Design command failed after all retry attempts.")[-4000:],
    )


def _prepare_proteinmpnn(request: DesignRequest) -> PreparedDesign:
    if request.target_path is None:
        return _configuration_required(request, "ProteinMPNN requires a target backbone structure.")
    repo = _env_path("MIRA_PROTEINMPNN_REPO") or _env_path("MIRA_PROTEINMPNN_PATH")
    script = repo / "protein_mpnn_run.py" if repo else None
    if not script or not script.exists():
        return _configuration_required(
            request,
            "ProteinMPNN is not installed. Set MIRA_PROTEINMPNN_REPO to the official ProteinMPNN checkout.",
        )

    target_path, target_metadata = _proteinmpnn_input_path(request)
    chain_string = _proteinmpnn_chain_string(request.chain_id)
    argv = [
        os.getenv("MIRA_PROTEINMPNN_PYTHON") or sys.executable,
        str(script),
        "--pdb_path",
        str(target_path),
        "--out_folder",
        str(request.output_dir),
        "--num_seq_per_target",
        str(request.num_designs),
        "--sampling_temp",
        str(request.temperature),
        "--batch_size",
        str(int(request.extra_args.get("batch_size") or 1)),
        "--seed",
        str(request.seed),
    ]
    if chain_string:
        argv.extend(["--pdb_path_chains", chain_string])
    weights = _env_path("MIRA_PROTEINMPNN_WEIGHTS")
    if weights:
        argv.extend(["--path_to_model_weights", str(weights)])
    model_name = os.getenv("MIRA_PROTEINMPNN_MODEL") or request.extra_args.get("model_name")
    if model_name:
        argv.extend(["--model_name", str(model_name)])
    if _truthy(os.getenv("MIRA_PROTEINMPNN_SOLUBLE")):
        argv.append("--use_soluble_model")

    fallbacks = []
    if chain_string and not _truthy(os.getenv("MIRA_PROTEINMPNN_DISABLE_FALLBACKS")):
        fallbacks.append(
            {"label": "retry_without_chain_selection", "argv": _argv_without_flag(argv, "--pdb_path_chains")}
        )

    return _prepared(
        request,
        argv=argv,
        backend="local",
        model="proteinmpnn",
        target_path=str(target_path),
        fallbacks=fallbacks,
        **target_metadata,
    )


def _prepare_ligandmpnn(request: DesignRequest) -> PreparedDesign:
    if request.target_path is None:
        return _configuration_required(request, "LigandMPNN requires a target backbone structure.")
    repo = _env_path("MIRA_LIGANDMPNN_REPO") or _env_path("MIRA_LIGANDMPNN_PATH")
    script = repo / "run.py" if repo else None
    if not script or not script.exists():
        return _configuration_required(
            request,
            "LigandMPNN is not installed. Set MIRA_LIGANDMPNN_REPO to the official LigandMPNN checkout.",
        )

    model_type = str(request.extra_args.get("model_type") or os.getenv("MIRA_LIGANDMPNN_MODEL_TYPE") or "protein_mpnn")
    argv = [
        os.getenv("MIRA_LIGANDMPNN_PYTHON") or sys.executable,
        str(script),
        "--pdb_path",
        str(request.target_path),
        "--out_folder",
        str(request.output_dir),
        "--model_type",
        model_type,
        "--number_of_batches",
        str(request.num_designs),
        "--seed",
        str(request.seed),
    ]
    temperature = str(request.temperature)
    if temperature:
        argv.extend(["--temperature", temperature])
    if request.chain_id:
        argv.extend(["--chains_to_design", request.chain_id.replace(" ", "").replace(",", " ")])
    for env_name, flag in [
        ("MIRA_LIGANDMPNN_CHECKPOINT", "--checkpoint_ligand_mpnn"),
        ("MIRA_LIGANDMPNN_PROTEIN_CHECKPOINT", "--checkpoint_protein_mpnn"),
    ]:
        checkpoint = _env_path(env_name)
        if checkpoint:
            argv.extend([flag, str(checkpoint)])

    return _prepared(request, argv=argv, backend="local", model="ligandmpnn")


def _prepare_foldingdiff(request: DesignRequest) -> PreparedDesign:
    length = _bounded_int(
        request.extra_args.get("length")
        or request.extra_args.get("backbone_length")
        or os.getenv("MIRA_FOLDINGDIFF_LENGTH"),
        default=80,
        low=int(os.getenv("MIRA_FOLDINGDIFF_MIN_LENGTH", "50")),
        high=int(os.getenv("MIRA_FOLDINGDIFF_MAX_LENGTH", "128")),
    )
    max_designs = int(os.getenv("MIRA_FOLDINGDIFF_MAX_DESIGNS", "8"))
    num_designs = max(1, min(request.num_designs, max_designs))
    batch_size = _bounded_int(
        request.extra_args.get("batch_size") or os.getenv("MIRA_FOLDINGDIFF_BATCH_SIZE"),
        default=max(num_designs, 8),
        low=1,
        high=512,
    )
    device = str(request.extra_args.get("device") or os.getenv("MIRA_FOLDINGDIFF_DEVICE") or "cpu")

    template = os.getenv("MIRA_FOLDINGDIFF_COMMAND") or os.getenv("MIRA_DESIGN_FOLDINGDIFF_COMMAND")
    if template:
        return _prepare_template_command(
            request,
            template,
            backend="local",
            model="foldingdiff",
            length=length,
            min_length=length,
            max_length=length + 1,
            batch_size=batch_size,
            device=device,
            num_designs=num_designs,
        )

    repo = _env_path("MIRA_FOLDINGDIFF_REPO") or _env_path("MIRA_FOLDINGDIFF_PATH")
    script = repo / "bin" / "sample.py" if repo else None
    if not script or not script.exists():
        return _configuration_required(
            request,
            "FoldingDiff is not installed. Set MIRA_FOLDINGDIFF_REPO to the FoldingDiff checkout "
            "and MIRA_FOLDINGDIFF_PYTHON to its Python environment.",
        )

    argv = [
        os.getenv("MIRA_FOLDINGDIFF_PYTHON") or sys.executable,
        str(script),
        "-l",
        str(length),
        str(length + 1),
        "-n",
        str(num_designs),
        "-b",
        str(batch_size),
        "--device",
        device,
        "--outdir",
        str(request.output_dir),
    ]
    if not _truthy(request.extra_args.get("run_psea")) and not _truthy(os.getenv("MIRA_FOLDINGDIFF_RUN_PSEA")):
        argv.append("--nopsea")
    if _truthy(request.extra_args.get("fullhistory")) or _truthy(os.getenv("MIRA_FOLDINGDIFF_FULL_HISTORY")):
        argv.append("--fullhistory")
    if _truthy(request.extra_args.get("testcomparison")):
        argv.append("--testcomparison")

    return _prepared(
        request,
        argv=argv,
        backend="local",
        model="foldingdiff",
        target_path=None,
        length=length,
        min_length=length,
        max_length=length + 1,
        batch_size=batch_size,
        device=device,
        num_designs=num_designs,
        output_kind="backbone_structure",
    )


def _prepare_rfdiffusion(request: DesignRequest) -> PreparedDesign:
    if request.target_path is None:
        return _configuration_required(request, "RFdiffusion requires a target/motif structure.", backend="gpu")
    template = os.getenv("MIRA_RFDIFFUSION_COMMAND") or os.getenv("MIRA_DESIGN_RFDIFFUSION_COMMAND")
    if template:
        return _prepare_template_command(request, template, backend="gpu", model="rfdiffusion")

    repo = _env_path("MIRA_RFDIFFUSION_REPO")
    script = repo / "scripts" / "run_inference.py" if repo else None
    contigs = request.extra_args.get("contigs") or os.getenv("MIRA_RFDIFFUSION_CONTIGS")
    if not script or not script.exists() or not contigs:
        return _configuration_required(
            request,
            "RFdiffusion requires a CUDA GPU worker plus MIRA_RFDIFFUSION_REPO and contigs "
            "(MIRA_RFDIFFUSION_CONTIGS or chat/tool args).",
            backend="gpu",
        )

    output_prefix = request.output_dir / "rfdiffusion"
    argv = [
        os.getenv("MIRA_RFDIFFUSION_PYTHON") or sys.executable,
        str(script),
        f"inference.input_pdb={request.target_path}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={request.num_designs}",
        f"contigmap.contigs={contigs}",
    ]
    hotspots = request.extra_args.get("hotspot_residues") or os.getenv("MIRA_RFDIFFUSION_HOTSPOTS")
    if hotspots:
        argv.append(f"ppi.hotspot_res={hotspots}")
    return _prepared(request, argv=argv, backend="gpu", model="rfdiffusion", contigs=contigs, hotspots=hotspots)


def _prepare_bindcraft(request: DesignRequest) -> PreparedDesign:
    if request.target_path is None:
        return _configuration_required(request, "BindCraft requires a target structure.", backend="gpu")
    template = os.getenv("MIRA_BINDCRAFT_COMMAND") or os.getenv("MIRA_DESIGN_BINDCRAFT_COMMAND")
    if template:
        return _prepare_template_command(request, template, backend="gpu", model="bindcraft")

    repo = _env_path("MIRA_BINDCRAFT_REPO")
    script = repo / "bindcraft.py" if repo else None
    settings = _env_path("MIRA_BINDCRAFT_SETTINGS")
    if not script or not script.exists() or not settings:
        return _configuration_required(
            request,
            "BindCraft requires a CUDA GPU worker plus MIRA_BINDCRAFT_REPO and MIRA_BINDCRAFT_SETTINGS.",
            backend="gpu",
        )

    argv = [
        os.getenv("MIRA_BINDCRAFT_PYTHON") or sys.executable,
        str(script),
        "--settings",
        str(settings),
        "--filters",
        os.getenv("MIRA_BINDCRAFT_FILTERS") or "default_filters",
        "--advanced",
        os.getenv("MIRA_BINDCRAFT_ADVANCED") or "default_4stage_multimer",
    ]
    return _prepared(request, argv=argv, backend="gpu", model="bindcraft", settings=str(settings))


def _prepare_custom(request: DesignRequest) -> PreparedDesign:
    template = os.getenv(f"MIRA_DESIGN_{request.library.upper()}_COMMAND") or os.getenv("MIRA_DESIGN_COMMAND")
    if not template:
        return _configuration_required(
            request,
            f"No real {request.library} design backend is configured.",
            backend="custom",
        )
    return _prepare_template_command(request, template, backend="custom", model=request.library)


def _prepare_template_command(
    request: DesignRequest, template: str, *, backend: str, model: str, **extra_format: Any
) -> PreparedDesign:
    target_path = "" if request.target_path is None else str(request.target_path)
    format_values = {
        "project_id": shlex.quote(request.project_id),
        "run_id": shlex.quote(request.run_id),
        "target_path": shlex.quote(target_path),
        "output_dir": shlex.quote(str(request.output_dir)),
        "chain_id": shlex.quote(request.chain_id),
        "num_designs": extra_format.get("num_designs", request.num_designs),
        "prompt": shlex.quote(request.prompt),
    }
    format_values.update({key: shlex.quote(str(value)) for key, value in extra_format.items()})
    command = template.format(
        **format_values,
    )
    parameters = _base_parameters(request, backend=backend, model=model)
    parameters.update({key: value for key, value in extra_format.items() if value is not None})
    parameters["shell_command"] = command
    return PreparedDesign(status="queued", command=command, parameters=parameters)


def _execution_attempts(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = [parameters]
    fallbacks = parameters.get("fallbacks") or []
    if isinstance(fallbacks, list):
        attempts.extend(item for item in fallbacks if isinstance(item, dict))
    return attempts


def _run_design_attempt(attempt: dict[str, Any], output_dir: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    argv = attempt.get("argv")
    shell_command = attempt.get("shell_command")
    if isinstance(argv, list) and argv:
        return subprocess.run(
            [str(item) for item in argv],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    if shell_command:
        return subprocess.run(
            str(shell_command),
            shell=True,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    raise ValueError("No executable design command is configured.")


def _attempt_command(attempt: dict[str, Any]) -> str:
    argv = attempt.get("argv")
    if isinstance(argv, list) and argv:
        return shlex.join([str(item) for item in argv])
    return str(attempt.get("shell_command") or "")


def _prepared(request: DesignRequest, *, argv: list[str], backend: str, model: str, **extra: Any) -> PreparedDesign:
    parameters = _base_parameters(request, backend=backend, model=model)
    parameters["argv"] = argv
    parameters.update({key: value for key, value in extra.items() if value is not None})
    return PreparedDesign(status="queued", command=shlex.join([str(item) for item in argv]), parameters=parameters)


def _configuration_required(request: DesignRequest, message: str, *, backend: str = "local") -> PreparedDesign:
    return PreparedDesign(
        status="configuration_required",
        command=None,
        parameters=_base_parameters(request, backend=backend, model=request.library),
        error=message,
    )


def _base_parameters(request: DesignRequest, *, backend: str, model: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "model": model,
        "target_path": str(request.target_path) if request.target_path is not None else None,
        "output_dir": str(request.output_dir),
        "chain_id": request.chain_id,
        "num_designs": request.num_designs,
        "temperature": request.temperature,
        "seed": request.seed,
        "prompt": request.prompt,
        "extra_args": request.extra_args,
    }


def _proteinmpnn_input_path(request: DesignRequest) -> tuple[Path, dict[str, Any]]:
    if request.target_path is None:
        raise ValueError("ProteinMPNN requires a target structure.")
    if request.target_path.suffix.lower() not in {".cif", ".mmcif"}:
        return request.target_path, {}
    prepared_path = request.output_dir / "proteinmpnn_input.pdb"
    try:
        from Bio.PDB import MMCIFParser, PDBIO

        structure = MMCIFParser(QUIET=True).get_structure(request.target_path.stem, str(request.target_path))
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(prepared_path))
    except Exception as exc:
        return request.target_path, {
            "source_target_path": str(request.target_path),
            "target_conversion_error": str(exc),
        }
    return (
        prepared_path,
        {
            "source_target_path": str(request.target_path),
            "prepared_target_path": str(prepared_path),
            "target_conversion": "mmcif_to_pdb",
        },
    )


def _proteinmpnn_chain_string(chain_id: str) -> str:
    raw = str(chain_id or "").strip()
    if not raw:
        return ""
    tokens = [
        token.strip(" .:-_[]{}")
        for token in re.split(r"[\s,;/]+", raw.replace("(", " ").replace(")", " "))
        if token and token.lower() not in {"chain", "chains", "and", "or"}
    ]
    tokens = [token for token in tokens if token]
    return " ".join(tokens)


def _argv_without_flag(argv: list[str], flag: str) -> list[str]:
    filtered = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            skip_next = True
            continue
        filtered.append(item)
    return filtered


def _bounded_int(value: object, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)


def _collect_sequences(output_dir: Path) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SEQUENCE_SUFFIXES:
            continue
        sequences.extend(_parse_fasta(path))
    return sequences[:1000]


def _parse_fasta(path: Path) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    header: str | None = None
    chunks: list[str] = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header and chunks:
                if not _is_source_sequence(path, header):
                    parsed.append(_sequence_record(path, header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header and chunks:
        if not _is_source_sequence(path, header):
            parsed.append(_sequence_record(path, header, "".join(chunks)))
    return parsed


def _is_source_sequence(path: Path, header: str) -> bool:
    first_token = header.split()[0].strip(",") if header else ""
    return first_token == path.stem


def _sequence_record(path: Path, header: str, sequence: str) -> dict[str, Any]:
    return {
        "id": header.split()[0] if header else path.stem,
        "header": header,
        "sequence": sequence,
        "length": len(sequence.replace("/", "")),
        "path": str(path),
        "file": path.name,
    }


def _collect_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif", ".fa", ".fasta", ".json", ".csv", ".txt", ".log"}:
            artifacts.append({"path": str(path), "file": path.name, "size_bytes": path.stat().st_size})
    return artifacts[:200]


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser()


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _tail_text(text: str, limit: int) -> str:
    return text[-limit:] if len(text) > limit else text
