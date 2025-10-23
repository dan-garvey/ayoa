# Changelog

## [Unreleased] - 2025-10-23

### Added
- **Auto-detect model from vLLM server**: The LLM client now automatically fetches the available model from the `/v1/models` endpoint
  - Queries the server on first use and caches the model name
  - Falls back to `MODEL_NAME` environment variable if server query fails
  - Prints auto-detected model name for transparency
  - `MODEL_NAME` in `.env` is now optional

### Changed
- **Renamed package from `gamecore` to `core`**:
  - All Python imports updated (`from gamecore.*` → `from core.*`)
  - Package name in `pyproject.toml` updated
  - CLI command remains `story` (no change for users)
  - Database filename changed from `gamecore.db` to `core.db`
  - All documentation updated to reflect new package name

### Fixed
- **Improved JSON parsing for LLM responses**:
  - Now strips `<think>` tags from chain-of-thought models (like Qwen3)
  - Better extraction of JSON from markdown code blocks
  - Handles JSON embedded in text responses
  - Regex-based extraction for more robust parsing
  - Applies to both narrative and structured JSON outputs

- **Clean narrative output**:
  - All `<think>` reasoning blocks are now automatically removed from story text
  - Improved error messages (truncated to 500 chars for readability)

### Technical Details
- Added `_fetch_model()` and `_ensure_model()` methods to `LLMClient`
- Model fetching happens lazily on first completion request
- Enhanced `complete()` and `complete_json()` methods with regex-based content cleaning
- Updated all 24 Python files with new import paths
- Updated configuration files: `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore`
- Updated documentation: `README.md`, `PROJECT_SUMMARY.md`

### Testing
- ✅ All 30 tests pass
- ✅ Tested with vLLM server (Qwen3-0.6B model)
- ✅ Story creation and continuation working properly
- ✅ JSON parsing robust with various model output formats

### Migration Notes
If you have an existing installation:
1. Remove old venv: `rm -rf .venv`
2. Create new venv: `uv venv --python 3.12 --seed` or `python3 -m venv .venv`
3. Activate and install: `source .venv/bin/activate && pip install -e ".[dev]"`
4. Update `.env` file - `MODEL_NAME` is now optional
5. Old save files will continue to work (JSON format unchanged)
6. Database will be recreated as `core.db` (old `gamecore.db` can be deleted)
