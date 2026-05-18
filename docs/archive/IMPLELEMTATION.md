## Implementation Sequence (7 Steps)

Each section is a **self-contained Claude Code prompt**. Copy directly. They build sequentially.

---

## Step 1: Project Scaffold + Tool Registry

**Give Claude Code:**

```
Create a Python project called `structagent` with this structure:

structagent/
├── pyproject.toml              # python 3.11, use uv
├── src/
│   └── structagent/
│       ├── __init__.py
│       ├── registry.py         # tool registration + schema generation
│       ├── tools/              # each tool is a separate module
│       │   ├── __init__.py     # imports all tool modules to trigger registration
│       │   ├── structure_io.py
│       │   ├── contacts.py
│       │   ├── sasa.py
│       │   ├── secondary_structure.py
│       │   ├── interface.py
│       │   └── alignment.py
│       ├── agent.py            # ReAct agent loop
│       ├── prompts.py          # system prompt templates
│       └── cli.py              # interactive chat interface
├── configs/
│   └── default.yaml            # model, max_steps, tool configs
├── tests/
│   └── test_registry.py
└── README.md

Implement registry.py with the following design (modeled after Hermes Agent's
registry pattern where tools self-register at import time):

class ToolRegistry (singleton):
    - _tools: dict mapping tool name -> ToolEntry

    @dataclass ToolEntry:
        name: str
        toolset: str          # logical grouping: "structure", "analysis", "ml"
        description: str
        parameters: dict      # JSON Schema for parameters
        handler: Callable     # the actual function
        check_fn: Optional[Callable[[], bool]]  # returns True if deps available

    Methods:
    - register(name, toolset, description, parameters, handler, check_fn=None)
      Called at module import time by each tool file.

    - get_tool_schemas(toolsets: Optional[List[str]] = None) -> list[dict]:
      Returns OpenAI-compatible function-calling tool schemas. Format:
      {"type": "function", "function": {"name": "load_structure",
       "description": "...", "parameters": {"type": "object",
       "properties": {...}, "required": [...]}}}
      If toolsets provided, filter. Exclude tools where check_fn() returns False.

    - call_tool(name: str, **kwargs) -> ToolResult:
      Dispatches to handler with error wrapping.

    - list_tools() -> list[str]: list available tool names

@dataclass
class ToolResult:
    success: bool
    data: str              # LLM-readable narrative text
    raw: dict              # machine-readable data for programmatic verification
    error: Optional[str] = None
    tool_name: str = ""
    execution_time_seconds: float = 0.0

Also create a convenience decorator:

def tool(name: str, toolset: str, description: str, parameters: dict):
    """Decorator that registers a function as an agent tool."""
    def decorator(func):
        registry.register(name=name, toolset=toolset, description=description,
                         parameters=parameters, handler=func)
        return func
    return decorator

The __init__.py in tools/ should import all tool modules so registration
happens on package import. For now, placeholder modules with pass.

Dependencies for pyproject.toml:
  biopython, gemmi, freesasa, requests, pyyaml, openai, scipy, numpy,
  rich (for CLI), click (for CLI entry point)

Note: we use the `openai` SDK because we target MiniMax M2.7's
OpenAI-compatible API at https://api.minimax.io/v1.

Write tests in test_registry.py that verify:
1. A decorated function gets registered
2. get_tool_schemas returns valid OpenAI function-calling format
3. call_tool dispatches correctly and wraps errors
4. check_fn filtering works (tool with failing check_fn excluded from schemas)
```

---

## Step 2: First Three Structural Tools

**Give Claude Code:**

```
Implement these three tools in src/structagent/tools/. Each uses the @tool
decorator from registry.py. All return ToolResult where `data` is
human-readable narrative and `raw` is machine-readable structured data.

DESIGN PRINCIPLE: The `data` field reads like a structural biologist's lab
notebook — complete sentences with context, not raw numbers. The LLM reasons
over `data`. The `raw` field stores exact values for programmatic verification.

### 1. structure_io.py — load_structure

@tool(
    name="load_structure",
    toolset="structure",
    description="Load a protein structure from the PDB by its 4-letter code. "
                "Returns metadata including resolution, chains, ligands, organism, "
                "and experimental method. Always call this first before using other "
                "structural analysis tools.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "4-letter PDB accession code (e.g., '1UBQ', '3ABC')"
            }
        },
        "required": ["pdb_id"]
    }
)

Implementation:
- Fetch mmCIF from https://files.rcsb.org/download/{pdb_id}.cif
- Parse with gemmi
- Cache files in ~/.cache/structagent/structures/
- Store parsed gemmi.Structure in module-level LRU dict (max 20)
- Export get_cached_structure(pdb_id) for other tools to import

data example:
"Structure 1UBQ: Ubiquitin, resolved at 1.80 Å by X-ray diffraction.
Contains 1 chain: A (76 residues, MET-1 to GLY-76).
No ligands. Organism: Homo sapiens.
Space group: P 21 21 21."

raw example:
{"pdb_id": "1UBQ", "resolution": 1.8, "method": "X-RAY DIFFRACTION",
 "chains": [{"id": "A", "length": 76, "first_residue": 1, "last_residue": 76,
             "entity_description": "Ubiquitin"}],
 "ligands": [], "organism": "Homo sapiens"}


### 2. contacts.py — get_residue_contacts

@tool(
    name="get_residue_contacts",
    toolset="structure",
    description="Find all residues making contacts with a target residue within "
                "a distance cutoff. Classifies contact types (salt bridge, hydrogen "
                "bond, hydrophobic, cation-pi, disulfide). Useful for understanding "
                "local interaction networks and key stabilizing contacts.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string", "description": "4-letter PDB code"},
            "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A')"},
            "residue_number": {"type": "integer", "description": "Residue sequence number"},
            "cutoff_angstroms": {
                "type": "number", "default": 4.5,
                "description": "Distance cutoff in Angstroms (default 4.5)"
            }
        },
        "required": ["pdb_id", "chain_id", "residue_number"]
    }
)

Implementation:
- Use get_cached_structure() from structure_io (auto-loads if not cached)
- Build scipy.spatial.KDTree over all heavy atom coordinates
- Find atoms within cutoff of any atom in target residue
- Group by residue, take min distance per pair
- Classify each contact:
    salt_bridge: (ARG NH1/NH2/NE or LYS NZ) within 4.0Å of (ASP OD1/OD2 or GLU OE1/OE2)
    hydrogen_bond: N/O donor within 3.5Å of N/O/S acceptor
    hydrophobic: both in {ALA,VAL,LEU,ILE,PHE,TRP,MET,PRO}, sidechain C within 4.0Å
    cation_pi: charged (ARG/LYS) within 6.0Å of aromatic centroid (PHE/TYR/TRP)
    disulfide: CYS SG within 2.5Å of CYS SG
    polar: other N/O/S contact within 3.5Å
    vdw: everything else
- Sort by distance

data example:
"Contacts for LYS-48 (chain A) in 1UBQ within 4.5 Å (8 contacts found):
  • GLN-49 (chain A): backbone hydrogen bond, 2.3 Å (N—O)
  • ALA-46 (chain A): hydrophobic contact, 3.8 Å (CB—CB)
  • GLU-51 (chain A): salt bridge, 3.1 Å (NZ—OE2)
  [...]"

raw: list of dicts with residue, chain, contact_type, distance, atom_pair


### 3. sasa.py — compute_sasa

@tool(
    name="compute_sasa",
    toolset="structure",
    description="Compute solvent-accessible surface area (SASA) for specified "
                "residues. Classifies as buried (<20% relative), partially "
                "exposed (20-50%), or exposed (>50%). Useful for identifying "
                "core vs surface residues and potential binding sites.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string", "description": "4-letter PDB code"},
            "chain_id": {"type": "string", "description": "Chain identifier"},
            "residue_range": {
                "type": "string",
                "description": "Residue range as 'start-end' (e.g., '45-52') or "
                              "comma-separated (e.g., '45,48,52'). Omit for entire chain."
            }
        },
        "required": ["pdb_id", "chain_id"]
    }
)

Implementation:
- Use freesasa library
- Convert gemmi structure to PDB string for freesasa input
- Compute relative SASA using Gly-X-Gly max values:
  ALA:129, ARG:274, ASN:195, ASP:193, CYS:167, GLN:225, GLU:223,
  GLY:104, HIS:224, ILE:197, LEU:201, LYS:236, MET:224, PHE:240,
  PRO:159, SER:155, THR:172, TRP:285, TYR:263, VAL:174
- Classify: buried (<20%), partial (20-50%), exposed (>50%)

data example:
"SASA for chain A, residues 1-10 in 1UBQ:
  MET-1:   162.3 Å² (68% relative, exposed)
  GLN-2:    54.1 Å² (24% relative, partially exposed)
  ILE-3:     8.7 Å² ( 4% relative, buried — core residue)
  [...]
Summary: 3 buried, 4 partially exposed, 3 exposed.
Mean relative SASA: 32.4%"


Write tests in tests/test_tools.py:
1. load_structure against 1UBQ
2. get_residue_contacts for LYS-48 in 1UBQ
3. compute_sasa for chain A residues 1-10 in 1UBQ
4. Error cases: bad PDB ID, nonexistent chain, nonexistent residue
5. ToolResult.success True/False correctness
6. raw data consistency with data narrative

Use @pytest.fixture to pre-load 1UBQ once.
```

---

## Step 3: Secondary Structure + Interface + Alignment Tools

**Give Claude Code:**

```
Implement three more tools building on structure_io caching from Step 2.

### 1. secondary_structure.py — get_secondary_structure

@tool(
    name="get_secondary_structure",
    toolset="structure",
    description="Assign secondary structure (helix, strand, coil) using DSSP. "
                "Returns per-residue assignments with element boundaries. "
                "Useful for identifying structural motifs and domain architecture.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string"},
            "chain_id": {"type": "string"},
            "residue_range": {
                "type": "string",
                "description": "Optional range ('start-end'). Omit for full chain."
            }
        },
        "required": ["pdb_id", "chain_id"]
    }
)

Implementation:
- gemmi.assign_dssp(structure), read residue.ss
- Map: H/G/I -> helix, E/B -> strand, else -> coil
- Group consecutive same-assignment residues into elements
- Report boundaries and lengths

data example:
"Secondary structure for chain A in 1UBQ:
  β-strand  1:  MET-1  — THR-7   (7 residues)
  coil      1:  LYS-8  — THR-12  (5 residues)
  β-strand  2:  ILE-13 — GLU-18  (6 residues)
  helix     1:  PRO-19 — GLU-24  (6 residues, α-helix)
  [...]
Summary: 5 β-strands, 2 helices, 7 coil regions."


### 2. interface.py — compute_interface

@tool(
    name="compute_interface",
    toolset="structure",
    description="Identify interface residues between two chains. Computes "
                "buried surface area, key contacts, and interface composition. "
                "Essential for protein-protein interaction analysis.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string"},
            "chain_a": {"type": "string", "description": "First chain ID"},
            "chain_b": {"type": "string", "description": "Second chain ID"},
            "distance_cutoff": {
                "type": "number", "default": 5.0,
                "description": "Distance cutoff for interface contacts (Å)"
            }
        },
        "required": ["pdb_id", "chain_a", "chain_b"]
    }
)

Implementation:
- Find cross-chain residue pairs within cutoff
- Buried SA = SASA(A alone) + SASA(B alone) - SASA(complex) via 3 freesasa runs
- Classify: % hydrophobic, % polar, % charged
- Hotspots: residues contributing >50 Å² buried SA

data example:
"Interface between chains A and B in 3HFM:
  Buried surface area: 1,847 Å² (A: 923 Å², B: 924 Å²)
  Interface residues: 24 on chain A, 26 on chain B
  Character: 45% hydrophobic, 35% polar, 20% charged

  Key interface residues (chain A, by buried SA):
    TRP-62:  142 Å² buried — central hydrophobic anchor
    ASP-101:  98 Å² buried — salt bridge with ARG-97(B)"


### 3. alignment.py — align_structures

@tool(
    name="align_structures",
    toolset="structure",
    description="Structurally align two protein chains and compute RMSD. "
                "Useful for comparing conformational states, assessing "
                "structural changes, and evaluating homolog similarity.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id_1": {"type": "string", "description": "First PDB code"},
            "chain_id_1": {"type": "string", "description": "Chain in first structure"},
            "pdb_id_2": {"type": "string", "description": "Second PDB code"},
            "chain_id_2": {"type": "string", "description": "Chain in second structure"}
        },
        "required": ["pdb_id_1", "chain_id_1", "pdb_id_2", "chain_id_2"]
    }
)

Implementation:
- gemmi.calculate_superposition on Cα atoms
- Overall RMSD + per-residue deviations
- Flag regions with >2Å deviation
- Report aligned length vs total

Test with:
- Secondary structure: 1UBQ chain A
- Interface: 3HFM or 1YCR
- Alignment: 1UBQ vs 1UBI
```

---

## Step 4: System Prompt

**Give Claude Code:**

```
Implement src/structagent/prompts.py.

def build_system_prompt(context: Optional[str] = None) -> str:
    """Build system prompt. Tool schemas go in the tools= parameter
    separately — NOT in the system prompt."""

SYSTEM_PROMPT = '''You are StructAgent, an expert structural biologist with deep
knowledge of protein structure, function, and dynamics. You analyze biomolecular
structures using computational tools to answer questions about protein mechanisms,
interactions, and allosteric regulation.

## Your Expertise
- Protein folding, stability, and structure-function relationships
- Enzyme mechanisms and active site architecture
- Protein-protein and protein-ligand interactions
- Allosteric regulation and signal propagation
- Post-translational modifications and their structural effects
- Evolutionary conservation and its structural implications

## How You Work
1. **Orient**: Load the structure first. Understand the protein, resolution,
   chains, ligands. Note experimental method and limitations.

2. **Hypothesize**: Form and state an explicit hypothesis about which
   structural features are relevant to the question.

3. **Investigate**: Use tools systematically:
   - Secondary structure to understand the fold
   - Contacts at key residues
   - SASA for buried vs exposed classification
   - Interface analysis if multi-chain
   - Alignment if comparing conformations

4. **Synthesize**: Integrate tool results with structural biology knowledge.
   Reference specific residues, distances, interaction types. Compare with
   known motifs.

5. **Qualify**: Note uncertainties, alternative explanations, limitations
   (static vs dynamic, crystal packing, resolution limits).

## Guidelines
- Cite residues with chain ID and number (e.g., "ARG-152 on chain A")
- Report distances from tool outputs — never fabricate numbers
- To trace allosteric pathways without a dedicated GNN tool, use sequential
  contact queries to follow the interaction network outward from a source site
- Cross-validate: check both contacts AND solvent accessibility for key residues
- Acknowledge when a question needs methods you lack (MD, mutagenesis data, etc.)
'''

EXAMPLE_QUERIES = {
    "allosteric_trace": (
        "Trace the interaction network from residue {source_residue} in {pdb_id}. "
        "Which residues relay structural information outward from this site?"
    ),
    "binding_interface": (
        "What are the key interactions stabilizing the interface between "
        "chains {chain_a} and {chain_b} in {pdb_id}?"
    ),
    "active_site": (
        "Identify and characterize the active site of {pdb_id}. "
        "What residues are catalytically important and why?"
    ),
    "stability": (
        "What structural features contribute to the stability of {pdb_id}? "
        "Identify key buried residues and interaction networks."
    ),
    "mutation_impact": (
        "Predict the structural impact of mutating {residue} in {pdb_id}. "
        "What interactions would be disrupted?"
    ),
}

Tests:
1. build_system_prompt() returns string with key sections
2. Context appended when provided
3. EXAMPLE_QUERIES format without errors
```

---

## Step 5: The Agent Loop

**Give Claude Code:**

```
Implement the ReAct agent loop in src/structagent/agent.py.

We use the OpenAI Python SDK pointing at MiniMax M2.7.

@dataclass
class AgentStep:
    thought: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[ToolResult]
    is_final: bool
    timestamp: float

@dataclass
class AgentRun:
    query: str
    steps: list[AgentStep]
    final_answer: str
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    wall_time_seconds: float
    model: str

    def to_dict(self) -> dict:
        """Serialize for saving trajectories to JSON."""

class StructAgent:
    def __init__(self,
                 model: str = "MiniMax-M2.7",
                 base_url: str = "https://api.minimax.io/v1",
                 max_steps: int = 15,
                 toolsets: Optional[list[str]] = None,
                 api_key: Optional[str] = None,
                 verbose: bool = True):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.environ.get("MINIMAX_API_KEY"),
            base_url=base_url
        )
        self.model = model
        self.max_steps = max_steps
        self.registry = get_registry()
        self.toolsets = toolsets
        self.verbose = verbose

    def run(self, query: str, context: Optional[str] = None) -> AgentRun:
        """Execute agent loop for a single query.

        Loop:
        1. Build system prompt via prompts.build_system_prompt(context)
        2. Get tool schemas from registry.get_tool_schemas(self.toolsets)
        3. messages = [
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": query}
           ]
        4. Loop up to max_steps:
           a. response = self.client.chat.completions.create(
                  model=self.model,
                  messages=messages,
                  tools=tool_schemas,
                  tool_choice="auto"
              )
           b. msg = response.choices[0].message
              finish = response.choices[0].finish_reason
           c. If msg.tool_calls is not None and len(msg.tool_calls) > 0:
              - Append assistant message to messages (include tool_calls)
              - For each tool_call:
                  name = tool_call.function.name
                  args = json.loads(tool_call.function.arguments)
                  result = registry.call_tool(name, **args)
                  Append: {"role": "tool", "tool_call_id": tool_call.id,
                           "content": result.data}
              - Log step, continue
           d. If finish == "stop" (no tool calls):
              - final_answer = msg.content
              - Break
        5. If max_steps exhausted, append user message:
           "You've used all tool calls. Synthesize findings into a final answer."
           Make one more call with tools=[] to force text response.
        6. Return AgentRun

        Token tracking: response.usage.prompt_tokens, .completion_tokens
        """

    def chat(self, messages: list[dict], context: Optional[str] = None) -> AgentRun:
        """Multi-turn: accepts full message history for conversational use."""

CRITICAL DETAILS:

1. OpenAI SDK tool format:
   tools = [{"type": "function", "function": {"name": "...",
             "description": "...", "parameters": {...}}}]

2. When the model returns tool_calls, you MUST append the full assistant
   message (with its tool_calls list) before appending tool results.
   The message sequence is:
     assistant (with tool_calls) → tool (result) → tool (result) → ...

3. Tool result messages use role="tool" and must include tool_call_id.

4. Multiple tool calls can happen in one response. Process ALL of them.

5. Parse tool_call.function.arguments with json.loads — it's a JSON string.

6. When verbose=True, use rich to print:
   🔧 load_structure(pdb_id="1UBQ")
   ├─ Resolution: 1.80 Å, 1 chain
   └─ ✓ 0.8s

   Final answer with rich.markdown rendering.
   Status line: "3 steps │ 2,847 tokens │ 14.2s"

7. Add 60-second timeout per tool call.

8. Wrap all LLM calls in try/except for API errors, rate limits,
   malformed responses. Log errors, return partial AgentRun.

Tests (mock the OpenAI client):
1. Simple 1-step (model returns text, no tool calls)
2. 2-step (model calls tool, gets result, responds)
3. max_steps enforcement
4. Error handling when tool fails
5. Multiple tool calls in single response
```

---

## Step 6: Interactive CLI

**Give Claude Code:**

```
Implement src/structagent/cli.py — interactive chat interface.

Use click for entry point, rich for terminal output.

@click.command()
@click.option("--model", default="MiniMax-M2.7", help="Model identifier")
@click.option("--base-url", default="https://api.minimax.io/v1", help="API base URL")
@click.option("--max-steps", default=15, help="Max tool-calling steps per query")
@click.option("--toolsets", default=None, help="Comma-separated toolsets to enable")
@click.option("--save-trajectories", is_flag=True, help="Save runs to JSONL")
@click.option("--trajectory-dir", default="./trajectories")
@click.option("--verbose/--quiet", default=True)
def chat(model, base_url, max_steps, toolsets, save_trajectories, trajectory_dir, verbose):
    """StructAgent — structural biology reasoning agent."""

Behavior:

1. Startup banner:
   ┌─────────────────────────────────────────┐
   │  StructAgent v0.1                       │
   │  Structural Biology Reasoning Agent     │
   │                                         │
   │  Model: MiniMax-M2.7                    │
   │  Tools: 6 available                     │
   │  Type /help for commands, /quit to exit │
   └─────────────────────────────────────────┘

2. List available tools on startup (name + one-line description)

3. REPL loop:
   Prompt: "You > "
   Commands:
     /tools     — list tools with descriptions
     /history   — conversation summary
     /clear     — reset conversation
     /save      — save current trajectory
     /example   — show example queries from prompts.EXAMPLE_QUERIES
     /quit      — exit

4. On each query:
   - Call agent.chat() with full message history (multi-turn)
   - Display tool calls as they happen (if verbose):

     🔧 load_structure(pdb_id="1DVY")
     ├─ Resolution: 2.00 Å, 2 chains (A, B)
     └─ ✓ 0.8s

   - Render final answer with rich Markdown
   - Status: "3 steps │ 2,847 tokens │ 14.2s"

5. Maintain conversation history across turns

6. If --save-trajectories: auto-save each AgentRun to
   {trajectory_dir}/{timestamp}.jsonl

7. Ctrl+C: interrupt current run, return to prompt
8. API errors: print, return to prompt

Entry point in pyproject.toml:
[project.scripts]
structagent = "structagent.cli:chat"

Usage:
$ structagent
$ structagent --model MiniMax-M2.7 --save-trajectories
$ structagent --base-url http://localhost:8000/v1 --model my-local-model

Test:
1. CLI starts without errors
2. /tools lists registered tools
3. /example shows queries
4. Mocked agent run displays correctly
```

---

## Step 7: Integration Test

**Give Claude Code:**

```
Run the full integration test for StructAgent.

Test query: "What structural features stabilize the hydrophobic core of
ubiquitin (1UBQ)? Identify the key buried residues and their interaction
networks."

Steps:

1. Verify all tools are registered:

   from structagent.tools import *  # triggers registration
   from structagent.registry import get_registry
   reg = get_registry()
   print(f"Registered tools: {reg.list_tools()}")
   # Should show: load_structure, get_residue_contacts, compute_sasa,
   #   get_secondary_structure, compute_interface, align_structures

2. Run the agent:

   from structagent.agent import StructAgent

   agent = StructAgent(
       model="MiniMax-M2.7",
       base_url="https://api.minimax.io/v1",
       verbose=True
   )
   run = agent.run(
       "What structural features stabilize the hydrophobic core of "
       "ubiquitin (1UBQ)? Identify the key buried residues and their "
       "interaction networks."
   )

   for i, step in enumerate(run.steps):
       print(f"\n--- Step {i+1} ---")
       if step.tool_name:
           print(f"Tool: {step.tool_name}({step.tool_args})")
           print(f"Result: {step.tool_result.data[:300]}...")
       if step.is_final:
           print(f"\nFinal Answer:\n{run.final_answer}")

   import json
   with open("test_run_1UBQ.json", "w") as f:
       json.dump(run.to_dict(), f, indent=2)

3. Quality checks:
   - Did the agent call load_structure first?
   - Did it use compute_sasa to find buried residues?
   - Did it use get_residue_contacts on buried residues?
   - Does the final answer reference specific residues and distances?
   - Does it acknowledge limitations (static structure)?

4. Also test the allosteric trace query:
   "Trace the interaction network from residue A:25 in 1DVY.
   Which residues relay structural information outward from this site?"

   This tests the agent's ability to do sequential contact queries
   to manually trace a pathway without a dedicated GNN tool.

5. Fix any tool errors. Common issues:
   - gemmi parsing edge cases for unusual mmCIF files
   - FreeSASA segfaults on large structures (add try/except)
   - KDTree memory on huge structures (cap atom count or use Cα-only)

Report what worked and what needs fixing.
```

---

## Architecture Summary

```
                       ┌─────────────┐
                       │   CLI Chat   │
                       │  (rich TUI)  │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │  StructAgent │
                       │  (ReAct Loop)│
                       │              │
                       │  OpenAI SDK  │
                       │  → MiniMax   │
                       │    M2.7      │
                       └──────┬───────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
        ┌──────▼──────┐ ┌────▼────┐ ┌───────▼───────┐
        │  Structure  │ │ Future  │ │   Analysis    │
        │  Toolset    │ │ ML/GNN  │ │   Toolset     │
        │             │ │ Tools   │ │               │
        │ load_struct │ │ (slot)  │ │ contacts      │
        │ sec_struct  │ │         │ │ sasa          │
        │ interface   │ └─────────┘ │ alignment     │
        └─────────────┘             └───────────────┘
               │
          ┌────▼────┐
          │ Registry │ (auto-discovery at import)
          └──────────┘
```

