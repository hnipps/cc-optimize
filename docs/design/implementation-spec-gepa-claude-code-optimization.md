# Implementation Spec: GEPA-Driven Claude Code Optimization MVP

## 1. Overview

Build a system that uses GEPA (Reflective Prompt Evolution) to optimize Claude Code's configuration surface for a single repository. The system runs benchmark tasks against candidate configurations, computes behavioral signals from execution traces, and uses GEPA's evolutionary loop to discover better configurations.

**Primary deliverable:** A Python package (`cc-optimize`) that a developer runs from the command line to optimize their Claude Code configuration for a given repository.

**Target runtime:** Python 3.10+, Linux/macOS, requires Claude Code CLI installed and authenticated (or use `claude-agent-sdk` which bundles the CLI).

---

## 1.1 Pre-Implementation Research Tasks

The following unknowns were identified through research but require hands-on investigation before implementation can begin. Complete these before starting Phase 2 (JSONL Parser) and Phase 7 (GEPA Integration).

### R1. Capture and document the Claude Code stream-json schema

The exact event structure from `--output-format stream-json` is not fully documented by Anthropic. From bug reports and community sources, the confirmed event types are:

```jsonc
{"type": "system", "subtype": "init", "session_id": "...", "tools": [...]}
{"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
{"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "...", "name": "Bash", "input": {...}}]}}
{"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}}
{"type": "result", "result": "...", "session_id": "...", "usage": {"input_tokens": N, "output_tokens": N}}
```

**Note:** Tool results come as `{"type": "user"}` events (not `{"type": "result"}`), and the `is_error` field is inside the `tool_result` content block. The `{"type": "result"}` event is only the final summary event. There is a known intermittent bug where the final `result` event is sometimes missing.

**Action:** Run these commands and save the output as test fixtures:
```bash
claude -p "Create a file called hello.txt containing 'Hello World'" \
  --output-format stream-json --verbose > tests/fixtures/sample_session.jsonl
claude -p "Run the command 'nonexistent-command' then run 'echo done'" \
  --output-format stream-json --verbose > tests/fixtures/error_cascade.jsonl
```
Then document every field of every event type before writing the parser.

### R2. Evaluate `claude-agent-sdk` vs raw CLI subprocess

Anthropic now provides `claude-agent-sdk` (pip installable, Python 3.10+) which wraps the CLI and provides typed Python objects (`AssistantMessage`, `ToolUseBlock`, etc.) via an async iterator. This may be a better integration path than raw subprocess + JSONL parsing.

**Action:** `pip install claude-agent-sdk`, then investigate:
- Do the message types expose `is_error` on tool results?
- How is token usage reported (per-message or only at end)?
- Can streaming be intercepted for early stopping?
- Is there enough structure to compute behavioral signals without raw JSONL?

**Decision point:** If the SDK provides all needed signal data, use it for the task runner and skip the JSONL parser entirely. If not, use raw CLI with JSONL parser. The spec is written assuming JSONL parsing (the more conservative option), but the SDK path should be evaluated first.

### R3. GEPA `objective_scores` and Pareto front behavior

GEPA's `EvaluationBatch` supports an `objective_scores` field: optional per-example maps of objective name → score. The spec uses three objectives (correctness, efficiency, conversation_quality).

**Action:** Read `src/gepa/core/adapter.py` and `src/gepa/core/state.py` to understand:
- Does `objective_scores` feed into Pareto front maintenance?
- Or is the Pareto front based only on per-example scalar `scores`?
- If multi-objective is supported natively, we can avoid collapsing to a weighted scalar.

### R4. GEPA reflective dataset schema expectations

The `make_reflective_dataset` method returns `dict[str, list[dict]]`. The default reflection proposer expects specific keys.

**Action:** Read `src/gepa/proposer/reflective_mutation/instruction_proposal.py` to determine:
- What keys does the default proposer read from each entry? (Confirmed at minimum: `"Inputs"`, `"Generated Outputs"`, `"Feedback"`)
- How much flexibility is there in the `"Feedback"` field?
- Should we use the default proposer or implement a custom `propose_new_texts()` for structured config mutations?

### R5. Claude Code permissions for headless benchmark runs

Every tool call in headless mode will prompt for permission unless explicitly allowed.

**Action:** Test these approaches and document which works:
- `--dangerously-skip-permissions` flag
- Permissive `settings.json`: `{"permissions": {"allow": ["Bash(*)", "Write(*)", "Edit(*)", "Read(*)", "MultiEdit(*)"]}}`
- `permission_mode="acceptEdits"` via the Python SDK

The chosen approach must be applied by the config applier (added to settings.json or passed as a CLI flag).

---

## 2. Architecture

```
cc-optimize/
├── cc_optimize/
│   ├── __init__.py
│   ├── cli.py                     # CLI entry point
│   ├── config.py                  # Global config / settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── benchmark.py           # Benchmark task data model
│   │   ├── candidate.py           # Configuration candidate data model
│   │   ├── signals.py             # Behavioral signal data model
│   │   └── result.py              # Evaluation result data model
│   ├── adapter/
│   │   ├── __init__.py
│   │   ├── gepa_adapter.py        # GEPA GEPAAdapter implementation
│   │   ├── config_applier.py      # Write candidate config to repo
│   │   ├── config_reader.py       # Read current repo config as seed
│   │   ├── task_runner.py         # Execute Claude Code CLI, capture JSONL
│   │   └── early_stopper.py       # Signal-based early termination
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── compute.py             # Main signal computation entry point
│   │   ├── efficiency.py          # Efficiency score from turn count
│   │   ├── repetition.py          # Bigram Jaccard on assistant blocks
│   │   ├── tool_errors.py         # Tool error cascade detection
│   │   ├── repair.py              # Repair/self-correction frequency
│   │   └── jsonl_parser.py        # Parse Claude Code JSONL output
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Orchestrate success + metrics + signals
│   │   ├── success_checker.py     # Run success criteria (tests, file checks)
│   │   ├── metrics.py             # Token count, turns, churn, error rate
│   │   └── llm_judge.py           # Optional LLM-as-judge scoring
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── loader.py              # Load benchmark tasks from YAML
│   │   ├── validator.py           # Validate benchmark definitions
│   │   └── baseline_runner.py     # Run baseline benchmarks with current config
│   └── optimization/
│       ├── __init__.py
│       ├── runner.py              # Orchestrate the full GEPA optimization run
│       ├── seed.py                # Generate seed candidate from current config
│       └── report.py              # Generate optimization report
├── benchmarks/                    # Benchmark task definitions (per-repo)
│   └── chainstats/
│       ├── suite.yaml             # Suite manifest
│       └── tasks/
│           ├── go-api-endpoint.yaml
│           ├── go-test-writing.yaml
│           └── ...
├── tests/
│   ├── unit/
│   │   ├── test_signals_efficiency.py
│   │   ├── test_signals_repetition.py
│   │   ├── test_signals_tool_errors.py
│   │   ├── test_signals_repair.py
│   │   ├── test_jsonl_parser.py
│   │   ├── test_config_applier.py
│   │   ├── test_config_reader.py
│   │   ├── test_success_checker.py
│   │   └── test_evaluator.py
│   ├── integration/
│   │   ├── test_task_runner.py
│   │   ├── test_gepa_adapter.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_session.jsonl     # Real CC session trace for signal tests
│       ├── looping_session.jsonl    # Session with repetition for signal tests
│       ├── error_cascade.jsonl      # Session with tool error cascades
│       └── sample_config/           # Sample repo config for applier tests
├── pyproject.toml
└── README.md
```

### Component Dependency Graph

Build order (each component only depends on components above it):

```
1. models/*              (zero dependencies, pure data)
2. signals/*             (depends on: models)
3. benchmark/loader      (depends on: models)
4. adapter/config_reader (depends on: models)
5. adapter/config_applier(depends on: models)
6. adapter/task_runner   (depends on: models, signals)
7. adapter/early_stopper (depends on: models, signals)
8. evaluation/*          (depends on: models, signals)
9. adapter/gepa_adapter  (depends on: everything above)
10. optimization/runner  (depends on: everything above + GEPA library)
11. cli.py               (depends on: everything)
```

---

## 3. Data Models

### 3.1 Benchmark Task (`models/benchmark.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaskCategory(Enum):
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DATABASE = "database"
    CROSS_STACK = "cross_stack"
    TEST_WRITING = "test_writing"
    DOCUMENTATION = "documentation"


@dataclass
class SignalBaselines:
    """Expected signal values for a task. Used to normalize scores
    and detect regressions."""
    efficiency_turn_baseline: int        # e.g. 8 for a bug fix, 15 for a feature
    max_looping_severity: int            # 0-3, usually 0 for well-scoped tasks
    max_tool_error_cascade: int          # e.g. 2
    max_repair_frequency: float          # e.g. 0.15


@dataclass
class SuccessCriterion:
    """A single verifiable check for task completion."""
    type: str
    # Supported types:
    #   "command_exit_code" — run a shell command, check exit code is 0
    #   "file_exists" — check a file path exists
    #   "file_contains" — check a file contains a substring or regex
    #   "file_not_contains" — check a file does NOT contain a string
    #   "git_diff_includes" — check that git diff includes changes to specific paths
    #   "git_diff_excludes" — check that git diff does NOT touch specific paths
    command: str | None = None           # For command_exit_code
    path: str | None = None              # For file_exists, file_contains
    pattern: str | None = None           # For file_contains, file_not_contains (regex)
    substring: str | None = None         # For file_contains, file_not_contains (literal)
    paths: list[str] | None = None       # For git_diff_includes/excludes (glob patterns)
    description: str = ""                # Human-readable description of what this checks


@dataclass
class BenchmarkTask:
    """A single benchmark task definition."""
    id: str                              # Unique task identifier, e.g. "go-api-endpoint"
    name: str                            # Human-readable name
    category: TaskCategory
    prompt: str                          # The task prompt sent to Claude Code
    repo_path: Path                      # Path to the target repository
    git_ref: str                         # Git commit/branch/tag for starting state
    success_criteria: list[SuccessCriterion]
    signal_baselines: SignalBaselines
    timeout_seconds: int = 600           # Max time for Claude Code to complete
    tags: list[str] = field(default_factory=list)  # e.g. ["go", "api", "handlers"]


@dataclass
class BenchmarkSuite:
    """Collection of benchmark tasks for a repository."""
    name: str                            # e.g. "chainstats"
    repo_path: Path
    tasks: list[BenchmarkTask]
    description: str = ""
```

### 3.2 Configuration Candidate (`models/candidate.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuleFile:
    """A single .claude/rules/*.md file."""
    filename: str                        # e.g. "go-conventions.md"
    content: str                         # Full file content including frontmatter
    paths: list[str] | None = None       # Parsed from frontmatter; None = unconditional


@dataclass
class SkillFile:
    """A skill directory with SKILL.md."""
    name: str                            # Skill directory name
    skill_md: str                        # SKILL.md content (description + instructions)


@dataclass
class ConfigCandidate:
    """A complete Claude Code configuration for a repository.
    This is the unit that GEPA optimizes."""
    id: str                              # Unique candidate identifier (UUID)
    claude_md: str                       # CLAUDE.md content
    rules: list[RuleFile] = field(default_factory=list)
    skills: list[SkillFile] = field(default_factory=list)
    settings_json: dict = field(default_factory=dict)  # .claude/settings.json overrides
    context_imports: list[str] = field(default_factory=list)  # @import paths in CLAUDE.md
    parent_id: str | None = None         # ID of the candidate this was mutated from
    mutation_rationale: str = ""         # GEPA's explanation of what changed and why

    def to_component_dict(self) -> dict[str, str]:
        """Convert to GEPA's native candidate format: dict[str, str] mapping
        component names to component text. GEPA evolves each component independently,
        selecting individual components for targeted mutation each iteration."""
        components = {}
        components["claude_md"] = self.claude_md
        for rule in self.rules:
            key = f"rule:{rule.filename}"
            # Embed path scope as a header comment so GEPA's reflection LLM can see it
            if rule.paths:
                header = f"# PATH_SCOPE: {json.dumps(rule.paths)}\n"
            else:
                header = "# PATH_SCOPE: unconditional\n"
            components[key] = header + rule.content
        for skill in self.skills:
            components[f"skill:{skill.name}"] = skill.skill_md
        if self.context_imports:
            components["context_imports"] = "\n".join(self.context_imports)
        # settings_json is NOT a GEPA component — it's structural config, not text to evolve
        return components

    @classmethod
    def from_component_dict(
        cls, components: dict[str, str], candidate_id: str,
        settings_json: dict | None = None,
    ) -> "ConfigCandidate":
        """Reconstruct from GEPA's component dict back to structured config."""
        claude_md = components.get("claude_md", "")
        rules = []
        skills = []
        context_imports = []
        for key, value in components.items():
            if key.startswith("rule:"):
                filename = key[len("rule:"):]
                # Parse the PATH_SCOPE header
                lines = value.split("\n", 1)
                paths = None
                content = value
                if lines[0].startswith("# PATH_SCOPE:"):
                    scope_str = lines[0][len("# PATH_SCOPE:"):].strip()
                    if scope_str != "unconditional":
                        paths = json.loads(scope_str)
                    content = lines[1] if len(lines) > 1 else ""
                rules.append(RuleFile(filename=filename, content=content, paths=paths))
            elif key.startswith("skill:"):
                name = key[len("skill:"):]
                skills.append(SkillFile(name=name, skill_md=value))
            elif key == "context_imports":
                context_imports = [line.strip() for line in value.strip().split("\n") if line.strip()]
        return cls(
            id=candidate_id,
            claude_md=claude_md,
            rules=rules,
            skills=skills,
            settings_json=settings_json or {},
            context_imports=context_imports,
        )
```

**GEPA multi-component design:** GEPA natively operates on candidates represented as `dict[str, str]` — a mapping from named components to text. In each optimization iteration, GEPA selects a *single component* to mutate based on its reflection analysis. This is a natural fit for Claude Code configuration:

- `"claude_md"` — the root CLAUDE.md file
- `"rule:go-conventions.md"` — each rule file is a separate component
- `"rule:api-design.md"` — GEPA can mutate one rule without touching others
- `"skill:questdb-patterns"` — each skill is independently evolvable
- `"context_imports"` — the list of @imported files

This means GEPA can discover optimizations like "the Go error handling rule needs different wording" without disturbing the Laravel conventions rule. It can also discover *structural* optimizations — for example, by proposing that content from `claude_md` should be moved into a new `rule:go-handlers.md` component. The reflection LLM sees all components and their PATH_SCOPE annotations, so it can reason about which component to improve.

**What GEPA cannot do natively:** GEPA mutates text within existing components but does not add/remove components from the dict. Adding a new rule file or splitting an existing one requires a custom `propose_new_texts()` implementation (see R4 in Pre-Implementation Research). For the MVP, start with fixed component structure (matching the seed config) and add structural mutations in a later iteration if needed.

**Round-trip invariant:** `ConfigCandidate.from_component_dict(c.to_component_dict(), c.id, c.settings_json)` must produce a `ConfigCandidate` that, when applied to a repo, produces identical files to applying `c` directly. Write a test for this.

### 3.3 Behavioral Signals (`models/signals.py`)

```python
from dataclasses import dataclass


@dataclass
class BehavioralSignals:
    """Deterministic behavioral metrics computed from a single session trace.
    Zero LLM cost — all computed from JSONL parsing."""

    # Efficiency: normalized turn count vs baseline
    # Formula: 1 / (1 + 0.3 * max(0, turns - baseline))
    # Range: 0.0-1.0, higher is better
    efficiency_score: float

    # Raw turn count (assistant messages)
    turn_count: int

    # Repetition: bigram Jaccard similarity between consecutive assistant blocks
    # Each pair of consecutive assistant text blocks is compared
    repetition_count: int                # Number of pairs with similarity >= 0.50
    repetition_exact_count: int          # Number of pairs with similarity >= 0.85
    repetition_max_severity: int         # 0=none, 1=minor(1-2 near-dupes), 2=moderate(3-5), 3=severe(6+)

    # Tool error cascade: max consecutive failed tool calls without an intervening success
    tool_error_max_cascade: int
    tool_error_total_failures: int
    tool_error_total_calls: int

    # Repair frequency: proportion of tool calls that were retries/corrections
    # A tool call is a "repair" if the previous tool call of the same type failed
    repair_frequency: float              # 0.0-1.0, lower is better
    repair_count: int

    # Session terminated early due to signal degradation?
    early_stopped: bool = False
    early_stop_reason: str = ""
```

### 3.4 Evaluation Result (`models/result.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskResult:
    """Result of running a single benchmark task with a given configuration."""
    task_id: str
    candidate_id: str
    success: bool                        # All success criteria passed?
    criteria_results: dict[str, bool]    # Per-criterion pass/fail
    signals: "BehavioralSignals"         # From models/signals.py
    tokens_input: int
    tokens_output: int
    tokens_total: int
    wall_time_seconds: float
    file_edit_churn: float               # edits to same file / distinct files edited
    tool_error_rate: float               # failed tool calls / total tool calls
    session_trace_path: Path             # Path to the raw JSONL
    llm_judge_scores: dict[str, float] = field(default_factory=dict)  # Optional


@dataclass
class CompositeScore:
    """Multi-objective score for GEPA's Pareto front."""
    correctness: float                   # 0.0-1.0 (fraction of criteria passed)
    efficiency: float                    # 0.0-1.0 (from behavioral signals)
    conversation_quality: float          # 0.0-1.0 (combined signal health)

    def weighted_scalar(self, weights: tuple[float, float, float] = (0.5, 0.25, 0.25)) -> float:
        """Collapse to a single scalar when needed. Default weights prioritize correctness."""
        w_c, w_e, w_q = weights
        return w_c * self.correctness + w_e * self.efficiency + w_q * self.conversation_quality


@dataclass
class MinibatchResult:
    """Result of running a candidate on a minibatch of tasks.
    This is what the GEPA adapter returns."""
    candidate_id: str
    task_results: list[TaskResult]
    composite_score: CompositeScore
    actionable_side_info: str            # Structured text for GEPA reflection LLM
```

**Conversation quality score** is derived from signals as follows:

```python
def compute_conversation_quality(signals: BehavioralSignals, baselines: SignalBaselines) -> float:
    """
    0.0-1.0 composite of signal health. Each signal contributes equally (0.25 weight).

    - Repetition component: 1.0 if severity <= baseline max, decreasing by 0.33 per severity level above
    - Tool error component: 1.0 if max_cascade <= baseline, then 1 / (1 + 0.5 * excess)
    - Repair component: 1.0 if frequency <= baseline max, then linear decrease to 0 at frequency=1.0
    - Efficiency component: the efficiency_score itself (already 0-1)
    """
```

---

## 4. Component Specifications

### 4.1 JSONL Parser (`signals/jsonl_parser.py`)

**Purpose:** Parse Claude Code's JSONL session output into structured events.

**Input:** Path to a `.jsonl` file produced by `claude -p "..." --output-format stream-json`.

**Claude Code JSONL format (confirmed from bug reports and community usage):** Each line is a JSON object. The event types are:

```jsonc
// Session initialization
{"type": "system", "subtype": "init", "session_id": "...", "tools": [...]}

// Assistant text message
{"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}

// Assistant tool use request (can be mixed with text in same content array)
{"type": "assistant", "message": {"content": [
  {"type": "text", "text": "I'll run that command..."},
  {"type": "tool_use", "id": "toolu_abc123", "name": "Bash", "input": {"command": "go test ./..."}}
]}}

// Tool result (comes as a "user" type event, NOT "result")
{"type": "user", "message": {"content": [
  {"type": "tool_result", "tool_use_id": "toolu_abc123", "content": "PASS\nok ...", "is_error": false}
]}}

// Final result with usage stats (only at end of session)
{"type": "result", "result": "...", "session_id": "...", "usage": {"input_tokens": N, "output_tokens": N}}
```

**Key schema observations from research:**
- Tool results are wrapped in `{"type": "user"}` events, not `{"type": "result"}` — the `"result"` type is reserved for the final session summary.
- A single assistant message can contain both `text` and `tool_use` content blocks in the same `content` array.
- The `is_error` field is on the `tool_result` content block inside the user message.
- There is a known intermittent bug where the final `{"type": "result"}` event is sometimes missing (the CLI process hangs). The parser must handle this gracefully — treat missing usage stats as zero rather than failing.
- With `--verbose` flag, additional `stream_event` events may appear for token-by-token streaming. These should be ignored.

**IMPORTANT:** The schema above is reconstructed from bug reports, community tools, and SDK source code — not from official documentation. **Complete research task R1** (capture real sessions) before implementing the parser to verify all field names and nesting.

**Output data structure:**

```python
@dataclass
class ToolCall:
    tool_use_id: str
    tool_name: str                # "Bash", "Write", "Edit", "Read", etc.
    input_data: dict
    output: str | None
    is_error: bool
    timestamp_index: int          # Ordering index in the session

@dataclass
class AssistantBlock:
    text: str
    tool_calls: list[ToolCall]    # Tool calls in this turn
    index: int                    # Turn index

@dataclass
class ParsedSession:
    assistant_blocks: list[AssistantBlock]
    all_tool_calls: list[ToolCall]
    total_input_tokens: int
    total_output_tokens: int
    raw_events: list[dict]        # All parsed JSON objects
```

**Implementation notes:**
- Read the file line by line, `json.loads()` each line
- Skip blank lines and lines that fail to parse (log a warning)
- Skip events with `type` not in `{"system", "assistant", "user", "result"}` (e.g., `stream_event`)
- Tool use events: extract from `assistant` events where content blocks have `type == "tool_use"`
- Tool result events: extract from `user` events where content blocks have `type == "tool_result"`. Match to tool_use by `tool_use_id`. The `is_error` field is on the `tool_result` block.
- A single assistant message can contain both text and tool_use blocks — extract both
- An assistant "turn" is defined as a contiguous block of assistant-type events before the next user event or end of session
- If the final `{"type": "result"}` event is missing, set total tokens to 0 and log a warning (known CLI bug)
- For headless (`-p`) runs, there is typically only one "user" turn (the prompt), followed by alternating assistant/user(tool_result) events

### 4.2 Signal Computation (`signals/`)

Each signal module takes a `ParsedSession` and returns its metric(s). The `compute.py` module orchestrates all of them.

#### 4.2.1 Efficiency (`signals/efficiency.py`)

```python
def compute_efficiency_score(session: ParsedSession, baseline_turns: int) -> tuple[float, int]:
    """
    Returns (efficiency_score, turn_count).

    efficiency_score = 1 / (1 + 0.3 * max(0, turn_count - baseline_turns))
    turn_count = len(session.assistant_blocks)
    """
```

**Test cases:**
| turn_count | baseline | expected score |
|-----------|----------|----------------|
| 15 | 15 | 1.0 |
| 25 | 15 | 1/(1+3.0) = 0.25 |
| 10 | 15 | 1.0 (cap, don't reward under-baseline) |
| 0 | 15 | 1.0 (edge case) |
| 16 | 15 | 1/(1+0.3) ≈ 0.769 |

#### 4.2.2 Repetition (`signals/repetition.py`)

```python
def bigram_jaccard(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between word bigram sets of two texts.

    1. Tokenize: re.findall(r'\b\w+\b', text.lower())
    2. Build bigram sets: {(words[i], words[i+1]) for i in range(len(words)-1)}
    3. Return |intersection| / |union| (0.0 if both empty)
    """

def compute_repetition(session: ParsedSession) -> tuple[int, int, int]:
    """
    Returns (repetition_count, exact_count, max_severity).

    For each consecutive pair of assistant text blocks (i, i+1):
    - Skip blocks with fewer than 5 words (too short for meaningful comparison)
    - Compute bigram_jaccard(block_i.text, block_{i+1}.text)
    - If similarity >= 0.85: exact_count += 1, repetition_count += 1
    - If 0.50 <= similarity < 0.85: repetition_count += 1

    Severity mapping:
    - 0: repetition_count == 0
    - 1: 1 <= repetition_count <= 2
    - 2: 3 <= repetition_count <= 5
    - 3: repetition_count >= 6
    """
```

**Test cases:**
| scenario | expected |
|----------|----------|
| Two identical long blocks | Jaccard=1.0, exact=1, count=1, severity=1 |
| Two completely unrelated blocks | Jaccard≈0.0, all counts=0, severity=0 |
| 7 consecutive near-identical blocks (6 pairs) | severity=3 |
| All blocks < 5 words | all counts=0, severity=0 (skipped) |
| Single block session | all counts=0, severity=0 |

#### 4.2.3 Tool Errors (`signals/tool_errors.py`)

```python
def compute_tool_error_cascade(session: ParsedSession) -> tuple[int, int, int]:
    """
    Returns (max_cascade, total_failures, total_calls).

    Walk session.all_tool_calls in timestamp_index order.
    Maintain current_cascade = 0.
    On failure: current_cascade += 1
    On success: record current_cascade, reset to 0.
    At end: record final current_cascade.

    max_cascade = max of all recorded cascade lengths.
    """
```

**Test cases:**
| tool call sequence (S=success, F=fail) | max_cascade |
|-----------------------------------------|-------------|
| S, S, S | 0 |
| F, S | 1 |
| F, F, F, F, F, S | 5 |
| S, F, F, S, F, S | 2 |
| F, F, F (no trailing success) | 3 |
| (empty) | 0 |

#### 4.2.4 Repair Frequency (`signals/repair.py`)

```python
def compute_repair_frequency(session: ParsedSession) -> tuple[float, int]:
    """
    Returns (repair_frequency, repair_count).

    A tool call at index i is a "repair" if:
    1. i > 0
    2. session.all_tool_calls[i-1].tool_name == session.all_tool_calls[i].tool_name
    3. session.all_tool_calls[i-1].is_error == True

    repair_count = number of repair tool calls
    repair_frequency = repair_count / len(all_tool_calls) (0.0 if empty)
    """
```

**Test cases:**
| sequence | repair_count | notes |
|----------|-------------|-------|
| Bash(fail), Bash(success) | 1 | Classic repair |
| Bash(fail), Write(success) | 0 | Different tool |
| Bash(fail), Bash(fail), Bash(success) | 2 | Both fail→same-tool transitions count |
| Bash(success), Bash(success) | 0 | No preceding failure |

#### 4.2.5 Compute Orchestrator (`signals/compute.py`)

```python
def compute_all_signals(session: ParsedSession, baselines: SignalBaselines) -> BehavioralSignals:
    """
    Compute all behavioral signals for a session.
    Calls each signal module and assembles into BehavioralSignals.
    """
```

### 4.3 Configuration Reader (`adapter/config_reader.py`)

**Purpose:** Read the current Claude Code configuration from a repository and produce a `ConfigCandidate`.

```python
def read_config(repo_path: Path) -> ConfigCandidate:
    """
    1. Read {repo_path}/CLAUDE.md → claude_md (empty string if missing)
    2. Read all .md files in {repo_path}/.claude/rules/
       - Parse YAML frontmatter for `paths:` field
       - Frontmatter delimited by --- on first line and --- on a subsequent line
       - Files without frontmatter or without paths: key → unconditional (paths=None)
    3. Read all subdirectories in {repo_path}/.claude/skills/
       - Each must contain a SKILL.md → skill_md
       - Directory name → skill name
    4. Read {repo_path}/.claude/settings.json → settings_json (empty dict if missing)
    5. Extract @import references from claude_md content
       - Pattern: lines matching ^@(\S+) → capture the path
    6. Generate UUID for candidate id
    7. Return ConfigCandidate with parent_id=None
    """
```

**Implementation notes:**
- Implement frontmatter parsing manually (split on `---` lines) to avoid external dependency. Only need to extract the `paths:` key, which is always a YAML list of strings.
- If `.claude/rules/` or `.claude/skills/` directories don't exist, return empty lists.
- `settings.json` may contain keys unrelated to optimization (e.g., API keys). Read the full dict; the applier will write it back as-is.

### 4.4 Configuration Applier (`adapter/config_applier.py`)

**Purpose:** Write a `ConfigCandidate` to a repository's filesystem.

```python
def apply_config(candidate: ConfigCandidate, repo_path: Path) -> None:
    """
    1. Verify repo_path exists and contains .git/
    2. Write {repo_path}/CLAUDE.md
    3. Delete all .md files in {repo_path}/.claude/rules/ (create dir if needed)
    4. Write each RuleFile:
       - If rule.paths is not None: prepend YAML frontmatter with paths
       - Write to {repo_path}/.claude/rules/{rule.filename}
    5. Delete all subdirectories in {repo_path}/.claude/skills/ (create dir if needed)
    6. For each SkillFile:
       - Create {repo_path}/.claude/skills/{skill.name}/
       - Write SKILL.md with skill.skill_md content
    7. Write {repo_path}/.claude/settings.json

    Raises ValueError if repo_path has no .git directory.
    Does NOT check for uncommitted changes (caller's responsibility).
    """
```

**Frontmatter generation for rules with paths:**
```python
def _generate_frontmatter(paths: list[str]) -> str:
    lines = ["---"]
    lines.append("paths:")
    for p in paths:
        lines.append(f'  - "{p}"')
    lines.append("---")
    return "\n".join(lines) + "\n"
```

### 4.5 Task Runner (`adapter/task_runner.py`)

**Purpose:** Execute a benchmark task via Claude Code CLI and capture the session trace.

```python
@dataclass
class RunResult:
    session_jsonl_path: Path
    exit_code: int
    wall_time_seconds: float
    timed_out: bool

def run_task(
    task: BenchmarkTask,
    candidate: ConfigCandidate,
    output_dir: Path,
    early_stop_config: EarlyStopConfig | None = None,
) -> RunResult:
    """
    Execution steps:

    1. Create a git worktree for isolation:
       git -C {task.repo_path} worktree add {worktree_path} {task.git_ref}
       where worktree_path = {output_dir}/worktrees/{task.id}_{timestamp}

    2. Apply the candidate config to the worktree:
       apply_config(candidate, worktree_path)

    3. Run Claude Code CLI:
       claude -p "{task.prompt}" --output-format stream-json --dangerously-skip-permissions
       Working directory: worktree_path
       Stdout → {output_dir}/traces/{task.id}_{candidate.id}_{timestamp}.jsonl
       Stderr → {output_dir}/traces/{task.id}_{candidate.id}_{timestamp}.stderr

       PERMISSIONS: Headless benchmark runs require unrestricted tool access.
       Use --dangerously-skip-permissions flag (appropriate since worktrees are
       isolated and disposable). If this flag is unavailable or insufficient,
       fall back to a permissive settings.json in the worktree:
       {"permissions": {"allow": ["Bash(*)", "Write(*)", "Edit(*)", "Read(*)", "MultiEdit(*)"]}}
       See pre-implementation research task R5 for which approach to use.

    4. If early_stop_config is provided:
       - Read stdout line by line
       - Every check_interval_events lines, parse partial session and check should_stop()
       - If should_stop() returns True, send SIGTERM to the process

    5. Wait for completion or timeout

    6. Do NOT remove the worktree yet (evaluator needs it for success checks)
       Return the worktree path inside RunResult so the caller can clean up.

    7. Return RunResult
    """
```

**Extended RunResult (add worktree_path):**
```python
@dataclass
class RunResult:
    session_jsonl_path: Path
    worktree_path: Path              # For success checking, cleaned up by caller
    exit_code: int
    wall_time_seconds: float
    timed_out: bool
    early_stopped: bool = False
    early_stop_reason: str = ""
```

**Fallback if worktrees fail:** Some repositories or git configurations don't support worktrees cleanly. If `git worktree add` fails, fall back to:
1. `git stash` (if dirty)
2. `git checkout --detach {git_ref}`
3. Run the task in the main repo
4. `git checkout -`
5. `git stash pop` (if stashed)

Log a warning when using the fallback.

### 4.6 Early Stopper (`adapter/early_stopper.py`)

```python
@dataclass
class EarlyStopConfig:
    max_looping_severity: int = 3
    max_tool_error_cascade: int = 5
    min_efficiency_score: float = 0.2
    check_interval_events: int = 10

def should_stop(
    partial_session: ParsedSession,
    baselines: SignalBaselines,
    config: EarlyStopConfig,
) -> tuple[bool, str]:
    """
    Compute signals on the partial session and check thresholds.

    Stop conditions (any one triggers):
    1. repetition_max_severity >= config.max_looping_severity
       → reason: "Repetition severity {N} reached threshold {M}"
    2. tool_error_max_cascade >= config.max_tool_error_cascade
       → reason: "Tool error cascade of {N} reached threshold {M}"
    3. efficiency_score < config.min_efficiency_score
       → reason: "Efficiency score {N:.2f} below threshold {M:.2f}"

    Returns (False, "") if no stop condition is met.
    """
```

### 4.7 Success Checker (`evaluation/success_checker.py`)

```python
def check_success(
    task: BenchmarkTask,
    worktree_path: Path,
) -> dict[str, bool]:
    """
    Run each SuccessCriterion against the post-task worktree state.
    Returns {criterion.description: pass/fail} for each criterion.

    Implementation per type:

    command_exit_code:
        result = subprocess.run(
            criterion.command, shell=True, cwd=worktree_path,
            capture_output=True, timeout=120
        )
        pass if result.returncode == 0

    file_exists:
        pass if (worktree_path / criterion.path).exists()

    file_contains:
        content = (worktree_path / criterion.path).read_text()
        if criterion.pattern: pass if re.search(criterion.pattern, content)
        if criterion.substring: pass if criterion.substring in content

    file_not_contains:
        content = (worktree_path / criterion.path).read_text()
        if criterion.pattern: pass if NOT re.search(criterion.pattern, content)
        if criterion.substring: pass if criterion.substring NOT in content

    git_diff_includes:
        changed = git diff --name-only {task.git_ref} (run in worktree)
        pass if ALL criterion.paths have at least one match in changed files
        (use fnmatch for glob matching)

    git_diff_excludes:
        changed = git diff --name-only {task.git_ref} (run in worktree)
        pass if NO criterion.paths match any changed files

    On any exception (file not found, command timeout): fail that criterion,
    log the error, continue checking remaining criteria.
    """
```

### 4.8 Metrics (`evaluation/metrics.py`)

```python
def compute_coding_metrics(session: ParsedSession) -> dict:
    """
    Returns {
        "tokens_input": int,
        "tokens_output": int,
        "tokens_total": int,
        "turn_count": int,
        "file_edit_churn": float,
        "tool_error_rate": float,
    }

    file_edit_churn:
        Examine all tool calls with tool_name in ("Write", "Edit", "MultiEdit").
        Extract the target file path from input_data.
        churn = total_write_edit_calls / distinct_files_targeted
        If no write/edit calls: churn = 0.0

    tool_error_rate:
        total_failures / total_tool_calls
        If no tool calls: 0.0
    """
```

### 4.9 Evaluator (`evaluation/evaluator.py`)

```python
def evaluate_task_run(
    task: BenchmarkTask,
    candidate: ConfigCandidate,
    run_result: RunResult,
) -> TaskResult:
    """
    1. Parse session: jsonl_parser.parse(run_result.session_jsonl_path)
    2. Compute signals: compute_all_signals(session, task.signal_baselines)
    3. Check success: check_success(task, run_result.worktree_path)
    4. Compute metrics: compute_coding_metrics(session)
    5. Assemble TaskResult
    """

def evaluate_minibatch(
    tasks: list[BenchmarkTask],
    candidate: ConfigCandidate,
    task_results: list[TaskResult],
    score_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> MinibatchResult:
    """
    1. Compute correctness: average of (criteria_passed / criteria_total) across tasks
    2. Compute efficiency: average of signals.efficiency_score across tasks
    3. Compute conversation_quality: average of compute_conversation_quality() across tasks
    4. Build CompositeScore
    5. Generate actionable_side_info (see format below)
    6. Return MinibatchResult
    """
```

### 4.10 GEPA Adapter (`adapter/gepa_adapter.py`)

**Purpose:** Implement the `GEPAAdapter` protocol that GEPA requires to plug into any system. This is the central integration point.

**GEPA's adapter interface (confirmed from source):** GEPA connects to external systems via the `GEPAAdapter` protocol defined in `src/gepa/core/adapter.py`. The adapter must implement two methods: `evaluate()` and `make_reflective_dataset()`. Optionally, a third method `propose_new_texts()` can override GEPA's default mutation logic.

GEPA represents candidates as `dict[str, str]` — a mapping from component names to component text. This maps directly to `ConfigCandidate.to_component_dict()`.

```python
import gepa
from gepa.core.adapter import EvaluationBatch

class ClaudeCodeAdapter(gepa.GEPAAdapter):
    """
    GEPA GEPAAdapter implementation for Claude Code configuration optimization.

    GEPA calls evaluate() with a candidate (dict[str, str]) and a batch of
    examples (benchmark tasks). We run each task via Claude Code CLI,
    compute behavioral signals, check success criteria, and return scores
    plus execution traces.
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        work_dir: Path,
        early_stop_config: EarlyStopConfig | None = None,
        settings_json: dict | None = None,  # Non-optimizable settings (permissions, etc.)
    ):
        self.suite = suite
        self.work_dir = work_dir
        self.early_stop_config = early_stop_config or EarlyStopConfig()
        self.settings_json = settings_json or {}
        (work_dir / "traces").mkdir(parents=True, exist_ok=True)
        (work_dir / "worktrees").mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        batch: list[dict],         # List of benchmark task dicts from trainset/valset
        candidate: dict[str, str], # Component name → component text
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """
        Called by GEPA to evaluate a candidate on a minibatch.

        Steps:
        1. config = ConfigCandidate.from_component_dict(candidate, uuid4(), self.settings_json)
        2. For each task_ref in batch:
           a. Look up the BenchmarkTask from self.suite
           b. Create git worktree at task.git_ref
           c. Apply config to worktree
           d. Run task via Claude Code CLI (with early stopping)
           e. Evaluate result (success criteria + signals + metrics)
           f. Clean up worktree
        3. Collect per-example scores and optional trajectories
        4. Return EvaluationBatch

        Returns:
            EvaluationBatch(
                outputs=[TaskResult, ...],     # Raw per-example outputs (GEPA doesn't interpret)
                scores=[float, ...],           # Per-example numeric scores for Pareto tracking
                trajectories=[dict, ...],      # Per-example traces (if capture_traces=True)
                objective_scores=[             # Optional multi-objective scores
                    {"correctness": 0.8, "efficiency": 0.9, "quality": 0.7},
                    ...
                ],
            )

        Score computation per example:
            correctness = criteria_passed / criteria_total
            efficiency = signals.efficiency_score
            quality = compute_conversation_quality(signals, baselines)
            score = 0.5 * correctness + 0.25 * efficiency + 0.25 * quality
        """

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """
        Called by GEPA after evaluate() to build a dataset for the reflection LLM.
        GEPA passes the list of component names it wants to improve.

        For each component in components_to_update, returns a list of dicts with:
        - "Inputs": description of the task and its context
        - "Generated Outputs": what Claude Code produced (summary, not full trace)
        - "Feedback": structured diagnostic text (the Actionable Side Information)

        The Feedback field is the most important — it's what GEPA's reflection LLM
        reads to propose mutations. See ASI format below.
        """
        reflective_data = {}
        for component in components_to_update:
            entries = []
            for task_result, score, trajectory in zip(
                eval_batch.outputs, eval_batch.scores, eval_batch.trajectories or []
            ):
                feedback = self._build_feedback(component, candidate, task_result, score)
                entries.append({
                    "Inputs": f"Task: {task_result.task_id}",
                    "Generated Outputs": self._summarize_output(task_result),
                    "Feedback": feedback,
                })
            reflective_data[component] = entries
        return reflective_data
```

**Feedback (ASI) generation per component:**

The `_build_feedback` method generates targeted diagnostic text for the specific component being considered for mutation. This is the key to GEPA's effectiveness — it tells the reflection LLM *why* the current component text is suboptimal.

```python
def _build_feedback(
    self,
    component_name: str,
    candidate: dict[str, str],
    task_result: TaskResult,
    score: float,
) -> str:
    """
    Generate component-specific feedback. Template:

    Score: {score:.2f} ({"PASS" if task_result.success else "FAIL"})

    ## Task: {task_result.task_id}
    Failed criteria: {list of failed criterion descriptions, or "None"}

    ## Signals vs baselines
    - Efficiency: {score:.2f} (turns: {actual}/{baseline})
    - Repetition: severity {sev} (threshold: {max})
    - Tool error cascade: {cascade} (threshold: {max})
    - Repair frequency: {freq:.2f} (threshold: {max})

    ## Component "{component_name}" context
    - This component is {line_count} lines / {word_count} words
    - {"This is CLAUDE.md, loaded every session" |
       "This rule has PATH_SCOPE: {paths}" |
       "This skill activates on description match"}
    - {"Was loaded for this task's file paths" | "Was NOT loaded for this task"}

    ## Trace observations relevant to this component
    - {specific observations from the execution trace that relate to this component}
    - {e.g., "Claude ignored the instruction on line 12 of this component to use
       structured error responses — it returned plain text errors instead"}
    """
```

**Component-specific observations:** When generating feedback for a specific component (e.g., `rule:go-conventions.md`), the feedback should focus on how that component's instructions affected the task. Scan the execution trace for:
1. Instructions in the component text that were clearly followed or ignored
2. Tool calls to files matching the component's PATH_SCOPE
3. Errors that relate to the domain covered by the component
4. Repetitive behavior on topics the component should have guided

### 4.11 Optimization Runner (`optimization/runner.py`)

```python
@dataclass
class OptimizationConfig:
    max_metric_calls: int = 150
    minibatch_size: int = 4
    stop_after_no_improvement: int = 20
    score_threshold: float = 0.95
    score_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    reflection_model: str = "anthropic/claude-sonnet-4-20250514"  # GEPA uses litellm format
    task_lm: str = "anthropic/claude-sonnet-4-20250514"          # For executing benchmark tasks
    work_dir: Path = Path("/tmp/cc-optimize")
    early_stop_config: EarlyStopConfig = field(default_factory=EarlyStopConfig)


def run_optimization(
    suite: BenchmarkSuite,
    seed_config: ConfigCandidate,
    config: OptimizationConfig,
) -> "OptimizationReport":
    """
    Run a GEPA optimization loop on the Claude Code configuration.

    The confirmed GEPA API (from gepa.optimize):

        import gepa

        result = gepa.optimize(
            seed_candidate=seed_dict,       # dict[str, str]
            trainset=trainset,              # list[dict] — benchmark tasks
            valset=valset,                  # list[dict] — same or subset for Pareto tracking
            adapter=adapter,                # GEPAAdapter instance
            task_lm=config.task_lm,         # LLM for task execution (not used by our adapter)
            reflection_lm=config.reflection_model,  # LLM for GEPA's reflection/mutation
            max_metric_calls=config.max_metric_calls,
        )

    Steps:
    1. Convert seed_config to component dict: seed_dict = seed_config.to_component_dict()
    2. Build trainset: list of dicts with task references
       trainset = [{"task_id": t.id} for t in suite.tasks]
       valset = trainset  # Same tasks for both train and validation in MVP
    3. Create the ClaudeCodeAdapter
    4. Call gepa.optimize()
    5. Extract results from the returned object:
       - result.best_candidate → dict[str, str]
       - result.candidates → list of all explored candidates
       - result.val_aggregate_scores → per-candidate scores
       - result.total_metric_calls → actual calls made
    6. Convert best candidate back: ConfigCandidate.from_component_dict(result.best_candidate, ...)
    7. Generate and return OptimizationReport

    NOTE on task_lm: GEPA's task_lm parameter is used by some adapters for the
    actual LLM calls during task execution. Our adapter runs Claude Code CLI directly,
    so task_lm is not used by the adapter — but GEPA may still require it as a parameter.
    If it's optional, omit it. If required, pass a dummy value.

    NOTE on trainset format: GEPA passes minibatches from trainset to adapter.evaluate().
    The adapter receives these as the `batch` parameter. Our adapter extracts task_id
    from each dict to look up the full BenchmarkTask from the suite. The trainset dicts
    can contain any fields — GEPA doesn't interpret them, just forwards to the adapter.
    """
```

**Result object:** GEPA returns an object with these confirmed fields (from the DSPy documentation):
- `best_candidate` — `dict[str, str]`, the best configuration found
- `candidates` — list of all proposed candidates
- `parents` — lineage info (parent indices for each candidate)
- `val_aggregate_scores` — per-candidate aggregate validation score
- `val_subscores` — per-candidate per-instance validation scores
- `per_val_instance_best_candidates` — per validation instance, which candidates scored best
- `discovery_eval_counts` — budget consumed to discover each candidate
- `total_metric_calls` — total evaluations performed

### 4.12 Report (`optimization/report.py`)

```python
@dataclass
class OptimizationReport:
    seed_candidate: ConfigCandidate
    best_candidate: ConfigCandidate
    pareto_front: list[ConfigCandidate]
    total_metric_calls: int
    total_tokens_consumed: int
    tokens_saved_by_early_stopping: int
    per_task_comparison: dict[str, dict]  # task_id → {"seed_score", "best_score", "delta"}
    signal_improvements: dict[str, dict]  # signal_name → {"before", "after", "delta"}
    optimization_trace: list[dict]        # Per-iteration: candidate_id, scores, mutation
    wall_time_seconds: float

    def to_markdown(self) -> str:
        """
        Generate human-readable report. Sections:
        1. Summary (seed score → best score, iterations, wall time, tokens)
        2. Per-task comparison table
        3. Signal improvement summary
        4. Top 3 most impactful mutations (from trace, by score delta)
        5. Best candidate configuration diff vs seed
        6. Pareto front summary (if multiple candidates)
        """

    def to_json(self) -> str:
        """Machine-readable JSON of all report data."""

    def save(self, output_dir: Path) -> None:
        """Write report.md, report.json, and best_candidate.yaml to output_dir."""
```

### 4.13 CLI (`cli.py`)

```python
"""
Commands:

  cc-optimize seed <repo_path> [--output PATH]
      Read current Claude Code config from repo, display summary,
      save as seed candidate YAML to --output (default: ./seed.yaml).

  cc-optimize baseline <suite_path> [--output-dir DIR]
      Run all tasks in the benchmark suite with the current repo config.
      Report per-task results with scores and signals.
      Save results to --output-dir (default: ./baseline/).

  cc-optimize optimize <suite_path> [--max-calls N] [--output-dir DIR] [--config PATH]
      Run GEPA optimization loop.
      --config: path to optimization config YAML (or use defaults).
      Save report and best candidate to --output-dir.

  cc-optimize apply <candidate_path> <repo_path>
      Apply a saved candidate YAML to a repository.
      Shows a diff preview and asks for confirmation before writing.

  cc-optimize report <output_dir>
      Display a previously generated optimization report.
"""
```

Use `click` library. Each command is a function decorated with `@click.command()`.

---

## 5. Benchmark Task YAML Schema

### suite.yaml

```yaml
name: chainstats
repo_path: /home/user/projects/chainstats
description: "ChainStats benchmark suite — Go/Laravel/Python multi-stack"
tasks:
  - tasks/go-api-endpoint.yaml
  - tasks/go-test-writing.yaml
  - tasks/laravel-controller.yaml
  - tasks/laravel-migration.yaml
  - tasks/python-analytics.yaml
  - tasks/cross-stack-feature.yaml
  - tasks/bug-fix-go.yaml
  - tasks/bug-fix-laravel.yaml
  - tasks/refactor.yaml
  - tasks/documentation.yaml
```

### Individual task YAML

```yaml
id: go-api-endpoint
name: "Add Go API endpoint for user preferences"
category: feature
prompt: |
  Add a new REST endpoint GET /api/v1/preferences/{user_id} in internal/handlers/.
  The handler should:
  - Validate that user_id is a valid UUID
  - Call the existing PreferenceService.GetByUserID method
  - Return a JSON response using the standard response envelope
  - Return 404 if the user has no preferences
  - Add appropriate tests in internal/handlers/preferences_test.go
git_ref: "benchmark/go-api-endpoint-start"
timeout_seconds: 600

signal_baselines:
  efficiency_turn_baseline: 15
  max_looping_severity: 0
  max_tool_error_cascade: 2
  max_repair_frequency: 0.15

success_criteria:
  - type: command_exit_code
    command: "go test ./internal/handlers/... -run TestPreferences -count=1"
    description: "Preferences handler tests pass"
  - type: command_exit_code
    command: "go build ./..."
    description: "Project compiles"
  - type: file_exists
    path: "internal/handlers/preferences.go"
    description: "Handler file created"
  - type: file_contains
    path: "internal/handlers/preferences.go"
    pattern: "func.*GetPreferences.*http\\.ResponseWriter"
    description: "Handler function has correct signature"
  - type: git_diff_excludes
    paths: ["go.mod", "go.sum"]
    description: "No dependency changes"

tags: [go, api, handlers, feature]
```

### Benchmark Loader (`benchmark/loader.py`)

```python
def load_suite(suite_path: Path) -> BenchmarkSuite:
    """
    1. Parse suite.yaml
    2. Resolve task file paths relative to suite.yaml's directory
    3. Load and validate each task YAML
    4. Return BenchmarkSuite

    Validation checks:
    - All task IDs are unique
    - All git_refs exist in the repo (git rev-parse --verify)
    - All task categories are valid enum values
    - All success criteria have required fields for their type
    - signal_baselines has all required fields
    """
```

---

## 6. Implementation Sequence

Build and test in this order. Each phase produces working, testable code.

### Phase 0: Project Scaffolding

**Tasks:**
1. Create directory structure as specified in Section 2
2. Create `pyproject.toml` with dependencies
3. Configure pytest
4. Create empty `__init__.py` files

**Acceptance:** `pytest` runs (0 tests collected, no errors).

**Dependencies (pyproject.toml):**
```toml
[project]
name = "cc-optimize"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pyyaml>=6.0",
    "click>=8.0",
    "gepa>=0.1.0",
]

[project.optional-dependencies]
sdk = [
    "claude-agent-sdk>=0.1.0",    # Alternative to raw CLI; evaluate in R2
]
dev = [
    "pytest>=7.0",
]

[project.scripts]
cc-optimize = "cc_optimize.cli:main"
```

### Phase 1: Data Models + Serialization

**Tasks:**
1. Implement all dataclasses in `models/`
2. Implement `ConfigCandidate.to_component_dict()` and `from_component_dict()` with round-trip test
3. Implement `benchmark/loader.py` with YAML parsing and validation
4. Write unit tests:
   - Component dict round-trip for ConfigCandidate (including PATH_SCOPE parsing)
   - YAML loading for BenchmarkTask
   - Validation error cases

**Acceptance:** All model tests pass. A sample task YAML loads correctly. ConfigCandidate survives component dict round-trip.

### Phase 2: JSONL Parser + Signal Computation

**Prerequisite:** Complete research task R1 (capture real Claude Code JSONL sessions) and R2 (evaluate whether `claude-agent-sdk` eliminates the need for raw JSONL parsing).

**Tasks:**
1. Capture 2-3 real Claude Code sessions as JSONL fixtures (or hand-craft realistic ones based on R1 findings):
   - `minimal_session.jsonl`: simple task, few tool calls, no errors
   - `looping_session.jsonl`: repeated assistant blocks with high similarity
   - `error_cascade.jsonl`: multiple consecutive tool failures
2. Implement `signals/jsonl_parser.py`
3. Implement each signal module: `efficiency.py`, `repetition.py`, `tool_errors.py`, `repair.py`
4. Implement `signals/compute.py`
5. Write unit tests for all signal modules against fixtures

**Acceptance:** `compute_all_signals()` returns correct BehavioralSignals for all 3 fixture files. All edge cases from Section 4.2 test tables pass.

### Phase 3: Config Reader + Applier

**Tasks:**
1. Create a test fixture: temp directory with `.git/`, `CLAUDE.md`, `.claude/rules/`, `.claude/skills/`
2. Implement `adapter/config_reader.py`
3. Implement `adapter/config_applier.py`
4. Write tests:
   - Read from fixture → apply to new temp dir → files match
   - Frontmatter parsing for path-scoped rules
   - Missing files/directories handled gracefully
   - Error on non-git directory

**Acceptance:** Round-trip test passes: `read_config(dir_a)` → `apply_config(candidate, dir_b)` → files in dir_b match dir_a.

### Phase 4: Task Runner + Early Stopper

**Tasks:**
1. Implement `adapter/early_stopper.py` (pure logic, no CLI dependency)
2. Implement `adapter/task_runner.py` with git worktree support and fallback
3. Write unit tests for early_stopper threshold logic
4. Write integration test: run a trivial Claude Code task, verify JSONL captured

**Acceptance:** A benchmark task executes, JSONL is captured to disk, and signals compute correctly from the captured output. Early stopper correctly identifies stop conditions on partial sessions.

**Note:** Integration tests in this phase require Claude Code CLI to be installed and authenticated. Mark them with `@pytest.mark.integration` so they can be skipped in CI.

### Phase 5: Evaluation Pipeline

**Tasks:**
1. Implement `evaluation/success_checker.py` with all criterion types
2. Implement `evaluation/metrics.py`
3. Implement `evaluation/evaluator.py` (evaluate_task_run + evaluate_minibatch)
4. Implement ASI generation function
5. Write tests:
   - Success checker against a temp repo with known state
   - Metrics computation from fixture sessions
   - Full evaluate_task_run pipeline (can use mocked RunResult)

**Acceptance:** `evaluate_task_run()` produces a correct TaskResult. `evaluate_minibatch()` produces a MinibatchResult with correct composite scores and well-structured ASI.

### Phase 6: Baseline Runner + CLI (subset)

**Tasks:**
1. Implement `benchmark/baseline_runner.py` — runs all tasks with current config, collects results
2. Implement CLI commands: `seed`, `baseline`
3. Run baseline against ChainStats (or a test repo)

**Acceptance:** `cc-optimize seed /path/to/repo` displays config summary and saves YAML. `cc-optimize baseline benchmarks/chainstats/suite.yaml` runs all tasks and produces a results table showing per-task pass/fail, scores, and signals.

**This phase validates the entire non-GEPA pipeline end-to-end.**

### Phase 7: GEPA Integration

**Prerequisite:** Complete research tasks R3 (objective_scores behavior) and R4 (reflective dataset schema).

**Tasks:**
1. Install GEPA: `pip install gepa`
2. Verify the API matches the spec's assumptions by running a minimal example:
   ```python
   import gepa
   result = gepa.optimize(
       seed_candidate={"test_prompt": "You are helpful."},
       trainset=[{"input": "hello"}],
       valset=[{"input": "hello"}],
       adapter=MinimalTestAdapter(),
       reflection_lm="anthropic/claude-sonnet-4-20250514",
       max_metric_calls=5,
   )
   print(type(result.best_candidate))  # Should be dict[str, str]
   ```
3. Implement `adapter/gepa_adapter.py` (`ClaudeCodeAdapter` with `evaluate` and `make_reflective_dataset`)
4. Implement `optimization/runner.py` using `gepa.optimize()`
5. Implement `optimization/report.py`
6. Implement CLI commands: `optimize`, `apply`, `report`
7. Integration test: short optimization run (5-10 metric calls)

**Acceptance:** `cc-optimize optimize suite.yaml --max-calls 10` completes, produces a report showing mutations attempted and score changes.

**Confirmed GEPA integration points:**
- Adapter: `ClaudeCodeAdapter(gepa.GEPAAdapter)` with `evaluate()` returning `gepa.EvaluationBatch` and `make_reflective_dataset()` returning `dict[str, list[dict]]`
- Runner: `gepa.optimize(seed_candidate=dict, trainset=list, adapter=adapter, reflection_lm=str, max_metric_calls=int)`
- Result: `result.best_candidate` (dict[str, str]), `result.total_metric_calls`, `result.val_aggregate_scores`, `result.candidates`

### Phase 8: Full Optimization + Validation

**Tasks:**
1. Create the full ChainStats benchmark suite (10 tasks with git refs and criteria)
2. Run full baseline
3. Run full optimization (150 metric calls)
4. Review report, apply best candidate
5. Re-run baseline with optimized config to confirm improvement
6. Use optimized config for real development work for 1 week
7. Document findings

**Acceptance:** MVP success criteria from the project doc (6/10 tasks improved, no regressions, at least one non-obvious optimization discovered, signals improved).

---

## 7. Testing Strategy

### Unit Tests (no external dependencies)
| Component | Key test cases |
|-----------|---------------|
| `models/candidate.py` | Round-trip via to_component_dict/from_component_dict; edge cases (empty rules, no skills, PATH_SCOPE parsing) |
| `signals/efficiency.py` | All rows from Section 4.2.1 test table |
| `signals/repetition.py` | Identical blocks, different blocks, short blocks, single block |
| `signals/tool_errors.py` | All rows from Section 4.2.3 test table |
| `signals/repair.py` | All rows from Section 4.2.4 test table |
| `signals/jsonl_parser.py` | Valid JSONL, malformed lines, missing fields, empty file |
| `adapter/config_reader.py` | Full config, partial config, missing dirs, frontmatter parsing |
| `adapter/config_applier.py` | Write + verify, error on non-git dir |
| `adapter/early_stopper.py` | Each threshold independently, combined thresholds |
| `evaluation/success_checker.py` | Each criterion type with pass and fail cases |
| `evaluation/metrics.py` | Token counting, churn calculation, error rate |
| `benchmark/loader.py` | Valid YAML, validation errors, missing files |

### Integration Tests (`@pytest.mark.integration`)
| Test | Requires |
|------|----------|
| Task runner captures JSONL | Claude Code CLI |
| End-to-end: run task → evaluate → TaskResult | Claude Code CLI, test repo |
| Baseline runner completes on 2 tasks | Claude Code CLI, test repo |

### Test Fixture Requirements
Complete research task R1 before writing the parser. Capture real JSONL:
```bash
# Minimal session
claude -p "Create a file called hello.txt containing 'Hello World'" \
  --output-format stream-json --verbose > tests/fixtures/sample_session.jsonl

# Session likely to have tool errors
claude -p "Run the command 'nonexistent-command' then run 'echo done'" \
  --output-format stream-json --verbose > tests/fixtures/error_cascade.jsonl
```

Document every event type and field observed in the captures. Update the JSONL parser section (4.1) if the actual schema differs from the confirmed-from-research schema above.

---

## 8. Global Configuration (`config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ClaudeCliConfig:
    command: str = "claude"
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class EarlyStopConfig:
    max_looping_severity: int = 3
    max_tool_error_cascade: int = 5
    min_efficiency_score: float = 0.2
    check_interval_events: int = 10


@dataclass
class OptimizationConfig:
    max_metric_calls: int = 150
    minibatch_size: int = 4
    stop_after_no_improvement: int = 20
    score_threshold: float = 0.95
    score_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    reflection_model: str = "anthropic/claude-sonnet-4-20250514"  # litellm format for GEPA


@dataclass
class GlobalConfig:
    work_dir: Path = Path("/tmp/cc-optimize")
    claude_cli: ClaudeCliConfig = field(default_factory=ClaudeCliConfig)
    early_stopping: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "GlobalConfig":
        """Load from YAML file. Falls back to defaults.
        Search order: explicit path → ./cc-optimize.yaml → ~/.cc-optimize/config.yaml"""
        ...

    def save(self, path: Path) -> None:
        """Save current config as YAML."""
        ...
```

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GEPA API details differ from spec assumptions | Adapter needs adjustment | Core API shape confirmed (`GEPAAdapter`, `evaluate`, `make_reflective_dataset`, `gepa.optimize()`). Remaining unknowns (R3, R4) are about Pareto internals, not the adapter interface. |
| Claude Code JSONL format varies by version | Parser breaks | Schema partially confirmed from bug reports. R1 captures real sessions before parser is written. Defensive parsing; log unknowns. Pin CC version. |
| `claude-agent-sdk` doesn't expose enough data for signals | Must use raw JSONL | R2 evaluates SDK before committing. Spec is written for JSONL path (conservative). Both paths share the same signal computation code. |
| Benchmark tasks too easy or too hard | No optimization signal | Phase 6 baseline run calibrates difficulty before GEPA runs. Adjust tasks and baselines. |
| Git worktrees fail for some repos | Task runner broken | Implement stash/checkout fallback. Config flag to select strategy. |
| 150 metric calls insufficient | Weak optimization | Check GEPA trace for ongoing improvement. Budget is adjustable. |
| GEPA cannot add/remove components (only mutate existing) | Limits structural optimization | MVP uses fixed component structure from seed config. Custom `propose_new_texts()` can be added later to enable structural changes (new rules, skill creation). |
| Claude Code headless permissions block tool calls | Tasks fail before starting | R5 tests permission approaches. Config applier adds permissive settings to worktree. |
| GEPA's reflection LLM doesn't understand Claude Code config semantics | Poor mutations | Rich `make_reflective_dataset` feedback with PATH_SCOPE annotations and signal diagnostics teaches the reflection LLM about config structure. |

---

## 10. Out of Scope

Explicitly excluded from this MVP:

- Outer loop / architecture exploration
- Harbor integration
- Multi-repository optimization
- Ecosystem monitoring automation
- V1 Collector integration
- CI/CD automation of optimization runs
- LLM-as-judge scoring (optional stretch goal, not required)
- OpenTelemetry signal export
- Dashboard / web visualization
- User frustration / positive feedback signals (need human interaction)
- Agent / sub-agent orchestration optimization (defer to second iteration after basic config optimization proves out)
