# ARC-AGI-3 Context for Agents

NOTE: The main ArcAGI playing agents do not read from this document.

> As of April 11, 2026. Everything below is current and verified against source code.

## What ARC-AGI-3 Is

A benchmark of ~25 interactive puzzle games played on a 64×64 grid. Each cell is an int 0-15 (a color). You send actions, get back the new grid. Goal: reach WIN state in fewest actions.

## The SDK (`arc-agi` 0.9.6)

```python
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
card_id = arcade.open_scorecard(tags=["my-agent"])
env = arcade.make("ls20", scorecard_id=card_id)

obs = env.reset()        # FrameDataRaw
obs = env.step(action, data={"x": 10, "y": 20}, reasoning={"thought": "..."})

scorecard = arcade.close_scorecard(card_id)
```

That's the entire API surface. Three calls: `reset()`, `step()`, `close_scorecard()`.

## Observation (`FrameDataRaw`)

Every `reset()` and `step()` returns:

| Field | Type | Meaning |
|-------|------|---------|
| `frame` | `list[ndarray]` | List of 64×64 int8 grids. `frame[-1]` is the latest. Values 0-15. |
| `state` | `GameState` | `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER` |
| `levels_completed` | `int` | How many levels cleared so far |
| `win_levels` | `int` | Total levels needed to win |
| `available_actions` | `list[int]` | IDs of legal actions right now |
| `guid` | `str` | Unique game instance ID |
| `game_id` | `str` | e.g. `"ls20"` |

## Actions

| Action | ID | Type | Meaning |
|--------|-----|------|---------|
| `RESET` | 0 | simple | Restart (level reset in competition mode) |
| `ACTION1` | 1 | simple | Up |
| `ACTION2` | 2 | simple | Down |
| `ACTION3` | 3 | simple | Left |
| `ACTION4` | 4 | simple | Right |
| `ACTION5` | 5 | simple | Interact / confirm (game-specific) |
| `ACTION6` | 6 | complex | Click at (x, y) where x,y ∈ [0,63]. Requires `data={"x": int, "y": int}` |
| `ACTION7` | 7 | simple | Undo |

`available_actions` tells you which are legal each step. ACTION6 availability doesn't specify which coordinates are valid — any (x,y) in range is accepted.

## The 16-Color Palette (universal, all games)

```
0=#FFFFFF white    4=#333333 off-black   8=#F93C31 red       12=#FF851B orange
1=#CCCCCC off-white 5=#000000 black      9=#1E93FF blue      13=#921231 maroon
2=#999999 gray-lt   6=#E53AA3 magenta   10=#88D8F1 blue-lt   14=#4FCC30 green
3=#666666 gray      7=#FF7BCC magenta-lt 11=#FFDC00 yellow   15=#A356D6 purple
```

The int value IS the color. What each color means is game-specific — the agent must figure it out from observation.

## Reasoning Field

`env.step(action, reasoning={"thought": "..."})` — opaque JSON blob, max 16KB. Stored in the recording. Use it to log chain-of-thought per step. Does not affect gameplay.

## Modes

| Mode | What it does | Use when |
|------|-------------|----------|
| `NORMAL` | Downloads game, runs locally. Free, no rate limits. | Developing |
| `ONLINE` | Hits three.arcprize.org API. Scorecard URL generated. | Testing scored runs |
| `COMPETITION` | Online + restrictions (see below). | Final submission |

### Competition Mode Restrictions
- One scorecard only
- One `make()` per game (can't restart a game)
- `RESET` becomes level reset (not game reset)
- `get_scorecard()` blocked mid-run
- All games scored even if you don't play them (zero score)

## Scoring

Per-level: `(human_baseline_actions / your_actions)²`

Fewer actions = higher score. Equal to baseline = 1.0. Twice baseline = 0.25. Aggregated across levels and games.

## Recordings

With `save_recording=True` on `arcade.make()`, each run saves a JSONL at `recordings/{card_id}/{game_id}-{guid}.jsonl`. Contains every frame, action, and reasoning blob. Replayable.

## Submission Flow

1. Run in `OperationMode.COMPETITION` (or `ONLINE` for unverified leaderboard)
2. Close scorecard → get `card_id`
3. Your scorecard URL: `https://arcprize.org/scorecards/{card_id}`
4. Fork https://github.com/arcprize/ARC-AGI-Community-Leaderboard
5. Add `submissions/your-id/submission.yaml`:
```yaml
name: "Your Method"
authors:
  - name: "Your Name"
code_url: "https://github.com/you/your-repo"
versions:
  - version: "1.0"
    date: 2026-04-11
    models:
      - name: "Gemini 2.5 Flash"
    scores:
      - benchmark: "arc-agi-3"
        scorecard_url: "https://arcprize.org/scorecards/your-card-id"
        set: "public"
```
6. Open PR. Maintainers merge weekly.

## Official Docs

Full docs: https://docs.arcprize.org/
Doc index for LLM consumption: https://docs.arcprize.org/llms.txt

Key pages:
- Toolkit overview: https://docs.arcprize.org/toolkit/overview
- Actions: https://docs.arcprize.org/actions
- Competition mode: https://docs.arcprize.org/toolkit/competition_mode
- Scoring methodology: https://docs.arcprize.org/methodology
- Available games: https://docs.arcprize.org/available-games
- Benchmarking: https://docs.arcprize.org/benchmarking-agent

SDK source: `pip show arc-agi` → installed at site-packages/arc_agi/
Engine source: `pip show arcengine` → installed at site-packages/arcengine/

## Current Preview Games

Three confirmed public games: `ls20` (Locksmith), `vc33`, `ft09`. Full list available via `arcade.get_environments()` or `GET /api/games`.


## Further Reading (fetch these URLs for deeper context)

### SDK & API
- https://docs.arcprize.org/llms.txt — full doc index, start here
- https://docs.arcprize.org/toolkit/overview — toolkit quickstart
- https://docs.arcprize.org/toolkit/arc_agi — Arcade class API
- https://docs.arcprize.org/toolkit/environment_wrapper — env.step/reset details
- https://docs.arcprize.org/toolkit/competition_mode — competition restrictions
- https://docs.arcprize.org/actions — full action space spec
- https://docs.arcprize.org/methodology — scoring formula (RHAE)
- https://docs.arcprize.org/available-games — list of all games
- https://docs.arcprize.org/rest_overview — REST API endpoints
- https://docs.arcprize.org/rate_limits — API rate limits
- https://docs.arcprize.org/scorecards — scorecard system
- https://docs.arcprize.org/vocabulary — ARC-AGI terminology

### Agent Building
- https://docs.arcprize.org/create-agent — how to build an agent
- https://docs.arcprize.org/agents-quickstart — quickstart guide
- https://docs.arcprize.org/llm_agents — LLM agent templates
- https://docs.arcprize.org/benchmarking-agent — benchmarking tooling
- https://docs.arcprize.org/toolkit/minimal — minimal code example
- https://docs.arcprize.org/game-schema — game file structure
- https://docs.arcprize.org/recordings — replay/recording format

### Submission
- https://github.com/arcprize/ARC-AGI-Community-Leaderboard — leaderboard repo
- https://github.com/arcprize/ARC-AGI-Community-Leaderboard/blob/main/CONTRIBUTING.md — submission rules
- https://github.com/arcprize/ARC-AGI-3-Agents — official agent templates

