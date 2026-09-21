"""Read-only public admission of the persisted eight-rank step-two checkpoint."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, '/data/CoordExp/.worktrees/coordexp-infras')
from src.artifacts.run_writer import admit_exact_resume_checkpoint_publication

checkpoint = Path('/data/CoordExp/outputs/infra_base/optimization-20260912/runs/candidate/checkpoints/step-2')
manifest_path = checkpoint / 'training_state/manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
run_path = checkpoint.parent.parent / 'run.json'
run_bytes = run_path.read_bytes()
receipt = {
    'checkpoint': str(checkpoint),
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'parent_run_sha256': hashlib.sha256(run_bytes).hexdigest(),
    'command': f'python {Path(__file__).resolve()} {sys.argv[1]}',
}
try:
    receipt['admitted'] = admit_exact_resume_checkpoint_publication(
        checkpoint,
        checkpoint_step=2,
        training_state_manifest_file_sha256=receipt['manifest_sha256'],
        training_state_aggregate_digest=manifest['aggregate_digest'],
        parent_run_id=manifest['parent_run_id'],
        parent_segment_id=manifest['parent_segment_id'],
    )
    status = 0
except Exception as error:
    receipt['failure_chain'] = []
    while error is not None:
        receipt['failure_chain'].append({'type': type(error).__name__, 'code': getattr(error, 'code', None), 'message': str(error)})
        error = error.__cause__
    status = 1
receipt['exit_code'] = status
assert manifest_path.read_bytes() == manifest_bytes
assert run_path.read_bytes() == run_bytes
receipt['parent_artifacts_unchanged'] = True
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, sort_keys=True))
raise SystemExit(status)
