# CC Optimize

Automated Claude Code configuration optimization using evolutionary search.

CC Optimize uses [GEPA](https://github.com/anthropics/gepa) (Guided Evolutionary Parameter Adaptation) to evolve your Claude Code configuration — CLAUDE.md, rules, skills, and settings — against a benchmark suite of real coding tasks, finding configurations that produce better results with fewer turns and fewer errors.

## How it works

1. **Seed** — Extract your current Claude Code config (CLAUDE.md, rules, skills, settings) from a repository
2. **Evaluate** — Run Claude Code against a benchmark suite of coding tasks, collecting behavioral signals (turn count, repetition, tool errors, repair frequency)
3. **Score** — Compute a weighted composite score from correctness (did the task succeed?), efficiency (how many turns?), and conversation quality (repetition, error cascades, repairs)
4. **Evolve** — GEPA uses an LLM-based reflection step to mutate the configuration, guided by per-task feedback on what went wrong
5. **Repeat** — Iterate until the score threshold is met, the budget is exhausted, or improvement stalls

Each task runs in an isolated git worktree. Sessions are streamed as JSONL for real-time early stopping when degenerate behavior is detected (looping, error cascades, efficiency collapse).

## Installation

Requires Python ≥ 3.10.

```bash
uv tool install cc-optimize
```

With optional dependencies:

```bash
uv tool install cc-optimize --with claude-agent-sdk   # Claude Agent SDK support
```

## Getting Started

### Prerequisites

- **Claude Code CLI** installed and authenticated (`claude` command available)
- A **target repository** with git initialized
- A **benchmark suite** — a set of coding tasks to evaluate against
- **API key for the reflection model** — the optimization loop uses an LLM to propose config mutations. By default this is `anthropic/claude-sonnet-4-20250514`, which requires an `ANTHROPIC_API_KEY` environment variable:

  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

  If you use a different provider (e.g. OpenAI), set the corresponding key (`OPENAI_API_KEY`, etc.) and change `reflection_model` in your config. See [Configuration](#configuration) for details.

### Step 1: Create a benchmark suite

Create a suite YAML file that references your target repo and a list of task files:

```yaml
# benchmarks/my-project/suite.yaml
name: my-project
repo_path: /path/to/my-project
description: "Benchmark tasks for my-project"
tasks:
  - tasks/fix-auth-bug.yaml
  - tasks/add-search-endpoint.yaml
  - tasks/refactor-db-layer.yaml
```

Each task file defines a prompt, a git ref to check out, signal baselines, and success criteria:

```yaml
# benchmarks/my-project/tasks/fix-auth-bug.yaml
id: fix-auth-bug
name: "Fix authentication token expiry bug"
category: bug_fix
prompt: |
  The JWT token refresh logic in src/auth/refresh.py silently swallows
  expiry errors. Fix it so expired tokens return a 401 response.
git_ref: "main"
timeout_seconds: 300

signal_baselines:
  efficiency_turn_baseline: 8
  max_looping_severity: 0
  max_tool_error_cascade: 2
  max_repair_frequency: 0.15

success_criteria:
  - type: command_exit_code
    command: "pytest tests/auth/ -x"
    description: "Auth tests pass"
  - type: file_contains
    path: "src/auth/refresh.py"
    pattern: "401"
    description: "Returns 401 on expiry"
```

### Step 2: Seed your current config

Extract your existing Claude Code configuration as a candidate YAML file:

```bash
cc-optimize seed /path/to/my-project
```

This reads CLAUDE.md, `.claude/rules/`, `.claude/skills/`, and `.claude/settings.json` from the repo and writes `seed.yaml`. Use `-o` to change the output path.

### Step 3: Run a baseline evaluation

See how your current config performs against the benchmark suite:

```bash
cc-optimize baseline benchmarks/my-project/suite.yaml
```

This runs every task with your current repo config and prints a results table with pass/fail status, efficiency score, turn count, and token usage per task. Results are saved to the `baseline/` directory (change with `-o`).

### Step 4: Run optimization

Launch the evolutionary optimization loop:

```bash
cc-optimize optimize benchmarks/my-project/suite.yaml
```

GEPA will iteratively mutate your config, evaluate candidates against task minibatches, and converge on higher-scoring configurations. The default budget is 150 metric calls.

Key options:

```bash
cc-optimize optimize suite.yaml --max-calls 200    # Increase evaluation budget
cc-optimize optimize suite.yaml --config cc-optimize.yaml  # Custom config
cc-optimize optimize suite.yaml -o results/         # Custom output directory
```

### Step 5: Review results

View the optimization report:

```bash
cc-optimize report optimization/
```

This displays the Markdown report generated during optimization, including the best score achieved, per-task breakdowns, and improvement over the seed.

### Step 6: Apply the best config

Apply the winning configuration to your repository:

```bash
cc-optimize apply optimization/best_candidate.yaml /path/to/my-project
```

This writes the optimized CLAUDE.md, rules, skills, and settings back to the repo. You'll be asked to confirm before any files are modified.

## Configuration

CC Optimize looks for configuration in this order:

1. Explicit `--config` flag
2. `./cc-optimize.yaml` in the current directory
3. `~/.cc-optimize/config.yaml` in your home directory

All fields are optional — missing values use sensible defaults:

```yaml
# cc-optimize.yaml
work_dir: /tmp/cc-optimize          # Working directory for worktrees and artifacts

claude_cli:
  command: claude                    # Claude Code CLI command
  extra_flags: []                   # Additional CLI flags

early_stopping:
  max_looping_severity: 3           # Repetition severity threshold (0-3)
  max_tool_error_cascade: 5         # Consecutive tool error threshold
  min_efficiency_score: 0.2         # Minimum efficiency before stopping
  check_interval_events: 10         # Check every N session events

optimization:
  max_metric_calls: 150             # Total evaluation budget
  minibatch_size: 4                 # Tasks sampled per evaluation
  stop_after_no_improvement: 20     # Stop if no improvement for N iterations
  score_threshold: 0.95             # Stop if score exceeds this
  score_weights: [0.5, 0.25, 0.25] # [correctness, efficiency, quality]
  reflection_model: "anthropic/claude-sonnet-4-20250514"
```

## CLI Reference

### `cc-optimize seed <repo_path>`

Extract the current Claude Code config from a repository.

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `seed.yaml` | Output path for candidate YAML |

### `cc-optimize baseline <suite_path>`

Run all benchmark tasks with the current config and report results.

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir` | `baseline` | Output directory for results |

### `cc-optimize optimize <suite_path>`

Run the GEPA optimization loop.

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --max-calls` | `150` | Max GEPA metric calls |
| `-o, --output-dir` | `optimization` | Output directory for results |
| `--config` | — | Path to optimization config YAML |

### `cc-optimize apply <candidate_path> <repo_path>`

Apply a candidate config to a repository. Requires confirmation.

### `cc-optimize report <output_dir>`

Display a previously generated optimization report.

## Benchmark Format

### Suite (`suite.yaml`)

```yaml
name: string          # Required — suite display name
repo_path: string     # Required — absolute path to target repo
description: string   # Optional — description
tasks:                # Required — list of relative paths to task files
  - tasks/task1.yaml
```

### Task (`tasks/*.yaml`)

```yaml
id: string                          # Required — unique identifier
name: string                        # Required — display name
category: string                    # Required — one of: bug_fix, feature, refactor,
                                    #   database, cross_stack, test_writing, documentation
prompt: string                      # Required — the prompt sent to Claude Code
git_ref: string                     # Required — git ref to checkout before running
timeout_seconds: int                # Optional — default 600

signal_baselines:                   # Required
  efficiency_turn_baseline: int     #   Expected turn count for this task
  max_looping_severity: int         #   Acceptable repetition severity (0-3)
  max_tool_error_cascade: int       #   Acceptable consecutive tool errors
  max_repair_frequency: float       #   Acceptable repair rate (0.0-1.0)

success_criteria:                   # Required — list of checks
  - type: command_exit_code         #   Run command, check exit code == 0
    command: string
  - type: file_exists               #   Check file exists
    path: string
  - type: file_contains             #   Check file matches pattern or substring
    path: string
    pattern: string                 #   Regex — or use `substring:` instead
  - type: file_not_contains         #   Check file does NOT match
    path: string
    pattern: string
  - type: git_diff_includes         #   All fnmatch patterns appear in git diff
    paths: [string]
  - type: git_diff_excludes         #   No fnmatch patterns appear in git diff
    paths: [string]

tags: [string]                      # Optional — metadata tags
```

## Key Concepts

### Signals

Behavioral signals are extracted from Claude Code session JSONL traces:

| Signal | Description |
|--------|-------------|
| `efficiency_score` | 0.0–1.0, penalizes excess turns beyond the baseline |
| `turn_count` | Number of assistant turns |
| `repetition_max_severity` | 0–3 severity of repeated output (bigram Jaccard similarity) |
| `tool_error_max_cascade` | Longest streak of consecutive tool failures |
| `repair_frequency` | Fraction of tool calls that are retries after failure |
| `early_stopped` | Whether the task was terminated early due to degenerate behavior |

### Scoring

Each candidate is scored on three dimensions:

- **Correctness** (default weight 0.5) — fraction of success criteria passed
- **Efficiency** (default weight 0.25) — penalizes excess turns beyond task baselines
- **Conversation Quality** (default weight 0.25) — composite of repetition health, tool error health, repair frequency health, and efficiency

Final score: `correctness × 0.5 + efficiency × 0.25 + quality × 0.25`

### GEPA

Guided Evolutionary Parameter Adaptation. The optimization algorithm represents your Claude Code config as a dictionary of named text components (CLAUDE.md, each rule, each skill, settings, imports). GEPA evaluates candidates against task minibatches, then uses an LLM reflection step to propose mutations based on per-task feedback — which signals were unhealthy, which criteria failed, and what the config components contained. This produces targeted, intelligent mutations rather than random perturbation.

## Development

```bash
# Clone and install
git clone <repo-url>
cd cc-optimize
uv tool install .

# Run tests
uv run pytest tests/unit/

# Project structure
cc_optimize/
├── cli.py                  # Click CLI entry point
├── config.py               # Global config dataclasses
├── models/                 # Data models (benchmark, candidate, result, signals)
├── signals/                # Signal computation (efficiency, repetition, tool errors, repair)
├── evaluation/             # Task evaluation, success checking, scoring
├── optimization/           # GEPA runner, seed generation, reporting
├── adapter/                # Claude Code integration (task runner, config reader/applier, early stopper)
└── benchmark/              # Suite/task loading and validation
```
