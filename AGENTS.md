# AGENTS.md

Guidance for coding agents working in this repository for the Global
Wheat track of the 5th BioHackathon Germany. 

## Environment

- Renku builds the session image from `project.toml` and `pyproject.toml`/`uv.lock`. `project.toml` installs the `pi`, `claude` and `codex` CLIs.
  `pyproject.toml` provides Python 3.12 with PyTorch, `transformers`, `datasets` and Streamlit.
- Everything needed is already installed. NEVER edit `project.toml`, `pyproject.toml`, `uv.lock` or
  `.venv/` unless the user asks. Changes there only apply once the session is rebuilt.
- GPUs depend on the session's resource class. Check `torch.cuda.is_available()` and fall back to
  `cpu` instead of assuming `cuda`.
- Files that aren't pushed to git are **lost** when the session is shut down. **Do not** write to a data connector.

## Repository layout

| Path | Purpose |
|---|---|
| `README.md` | Hackathon overview  |
| `LICENSE` | Repository license |
| `docs/CONTRIBUTING.md` | Contribution rules, AI policy, liability disclaimer |
| `examples/` | a Streamlit app and Jupyter notebooks that load Global Wheat data, run HuggingFace models and call the external API; plus a small sample dataset and demo GIFs |
| `project.toml`, `pyproject.toml`, `uv.lock` | Renku session image and Python environment (see Environment) |
| `.githooks/pre-commit` | Blocks commits that contain secrets, tokens or credential files |
| `init.d/install-git-hooks.sh` | Runs at Renku session start; links `.githooks/pre-commit` into each repo |

## Workflow 1: Renku only (default)

- **Dataset:** the GWHS dataset, available via HuggingFace (will be published soon) from [the GlobalWheat HF account](https://huggingface.co/GlobalWheat).
- **Models:** available from HuggingFace, loaded with `transformers`. 
- **Public sample images:** `https://data.global-wheat.com/{gwhd,gwfss}/images/00000000.{jpg,png}`.

## Serving web apps from a session

The session sits behind Renku's proxy - the session ID. It can be bound to localhost `127.0.0.1` and base path can be retrieved with
`$RENKU_BASE_URL_PATH`. Example for Streamlit:

```sh
--server.baseUrlPath="${RENKU_BASE_URL_PATH%/}/proxy/absolute/8501"
```

The app then opens at `https://<session host>${RENKU_BASE_URL_PATH%/}/proxy/absolute/8501/`.
The user may then stop it at will with Ctrl+C.

## Workflow 2: the collaborator's API (optional, external)

- Dr. Hanzhou Wang built a gateway for this event:
  <https://github.com/UTokyo-FieldPhenomics-Lab/gwhs-biohackathon-api-gateway>. Participant docs
  are at <https://docs.global-wheat.com>. It works both on Renku and on the participant's own machine.
- The base URL is `https://api.global-wheat.com`. `/health`, `/docs` and `/openapi.json` are public.
  Every other path needs the request header `X-API-Key: <team token>`. Participants receive tokens on (or
  before) December 7.
- The dataset ids are `gwhd`, `gwfss`, `gwhs-temp` (a development sample) and `gwhs` once it is
  released. Image bytes are served from `data.global-wheat.com`.
- Read the key from the environment (`GLOBAL_WHEAT_API_KEY`). If it's not in the environment, prompt the user:
it should be available as an envvar via Renku (Session -> Secrets) or on their own machine.
  **NEVER** hard-code, print, log or commit this or any other secret key.

> **Disclaimer:** SDSC does not maintain the API gateway, its clients or its documentation. The SDSC and the maintainers of this repository accept **no
> responsibility** for any damage, data loss, credential leakage or security incident caused by
> using it or by irresponsible use of AI agents. Review anything you take from it. When unsure, use Workflow 1.

## Rules

- Follow the [SDSC LLM-assistant guidelines](https://swissdatasciencecenter.github.io/best-practice-documentation/docs/chapters/security/llm-assistant/)
  and `docs/CONTRIBUTING.md`. 
- **Do not** read, print or commit secrets: API keys, HuggingFace tokens, `.env` files, agent credentials.
- Don't commit or push unless deliberately asked. A human should review every change.
- Don't add large files to git (images, weights, datasets). 
- Oblige by the githooks preconfigured in this repository. Never skip them.
