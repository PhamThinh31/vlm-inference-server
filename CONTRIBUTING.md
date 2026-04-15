# Contributing

## Dev environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

## Before you open a PR

```bash
make lint       # ruff + mypy
make test       # unit + integration
```

Both must pass. CI runs the same commands.

## House rules

These are not style preferences; they are load-bearing:

1. **No sentinel defaults.** Do not introduce `category="unknown"`,
   `confidence=0.5`, `is_valid=False` as a stand-in for "we don't know."
   Use `Optional[T] = None` + an explicit `error` field. See
   `tests/unit/test_pipeline_contracts.py` — those tests guard the
   contract and must stay green.

2. **No positional fallbacks.** Matching a model response to a request
   by list index is a silent data-corruption vector. Match by ID or
   mark the row `parse_failed=True`.

3. **No `try / except: log-and-pass` at API boundaries.** Either the
   caller needs to know (surface it in the response envelope) or the
   exception is truly benign (narrow the exception type, document why).

4. **Blocking work on the event loop is a bug.** PIL decode, JSON
   parse of large responses, file I/O, model init — all go through
   `asyncio.to_thread` or a bounded executor. Review your diff for new
   `def _foo()` synchronous functions called inside `async def`.

5. **Every new env var gets a default in `app/config.py`.** No
   `os.getenv("X")` scattered through the code.

6. **New tests for new parsers.** If you add a `_parse_response` path,
   add adversarial fixtures in `tests/unit/test_pipeline_contracts.py`.

## Commit messages

Subject line: imperative, <= 72 chars.
Body (optional): what changed, *why*, what you considered and rejected.
Do not reference Claude, Copilot, issue numbers, or PR IDs — they rot.

## Scope discipline

Small PRs merge. Large PRs rot. If your diff touches more than ~400
lines across unrelated areas, split it. Refactors and feature work do
not share a PR.
