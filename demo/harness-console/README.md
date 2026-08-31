# TongAgent Harness Console

An offline, read-only visualization for recording the final TongAgent demo.
It replays three frozen formal artifacts—a direct success, a run that revises
its search query, and a safe budget stop—plus one frozen controlled
checkpoint-recovery artifact. It never calls a model or the public network.

## Generate the sanitized bundle

From the repository root:

```bash
python demo/harness-console/generate_demo_bundle.py
python demo/harness-console/generate_demo_bundle.py --check
```

The generator intentionally excludes the reference answer, credentials,
provider base URLs, and page body text. It fails if the chosen formal artifact
does not satisfy `raw_model_answer == final_answer`.

## Run locally

```bash
./demo/harness-console/run_demo.sh
```

Open <http://127.0.0.1:8765> in a browser. The site is designed for a 1920×1080
recording canvas. Select one of the four traces, then use automatic replay or
single-step mode. Each event identifies its real code module and function; the
canonical artifact directory is displayed above the trace.

When the repository is on a remote server, forward port `8765` through SSH or
the IDE's Ports panel. Keep the server bound to `127.0.0.1`; no public bind is
needed for recording.

## Provenance

- Formal artifact replay:
  `final-transparent-harness-v2-formal-20260722T162330Z`
- Replay tasks:
  `0383a3ee-47a7-41a4-b493-519bdefe0488`,
  `305ac316-eef6-4446-960a-92d80d542f82`,
  `46719c30-f4c3-4cad-be07-d5cb21eee6bb`
- Controlled fault experiment:
  `final-transparent-harness-v2-fault-20260722T175204Z`
- Controlled checkpoint-recovery task:
  `fault-resume-alpha`

The visualization is a presentation surface, not a new Agent runtime. Canonical
results remain under `learning/search-agent/output/evaluations/`.
