# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- The repo's `.mcp.json` registers the iDEA MCP server as `idea` (it runs `python -m iDEA.mcp_server`, so
  install with `pip install -e ".[dev,mcp]"` first). Approve it when prompted; its tools then appear as
  `mcp__idea__*`.
- Use the MCP tools for quick physics questions and exploration (energies, densities, Kohn-Sham potentials,
  plots). Write Python against the iDEA API directly for scripts, notebooks, parameter sweeps and anything that
  should be reproducible or committed.
- Keep exact (`interacting`) calculations small while exploring (see "Cost of the exact solver" above); the
  server refuses many-body dimensions above 5e6 unless `allow_large=True`.
- When changing library code, run `ruff check`, `ruff format` and `pytest -v -k "not Long"` before committing.
  If you change a function the MCP server wraps, also run `pytest tests/test_mcp_server.py`.
