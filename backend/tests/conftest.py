import os
import tempfile

# Keep run persistence out of the repo during tests; must be set before app import.
os.environ.setdefault("PULSEGRAPH_RUNS_DIR", tempfile.mkdtemp(prefix="pulsegraph-test-runs-"))
