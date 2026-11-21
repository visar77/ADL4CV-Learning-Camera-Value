"""Persist object skips across allocations of a prepared batch."""

import os


class ObjectSkips:
    def __init__(self, run_root, args, load_json, save_json):
        self.path = run_root / "skipped_objects.json"
        self.enabled = (
            args.skip_failed_object
            and not (args.object_id or args.experiment_id or args.candidate_id)
            and (run_root / "run_manifest.json").is_file()
        )
        self.records = load_json(self.path) if self.enabled and self.path.is_file() else {}
        self.save_json = save_json

    def should_skip(self, object_id):
        if object_id not in self.records:
            return False
        print(f"Skipping {object_id} from {self.path.name}", flush=True)
        return True

    def record(self, object_id, experiment_id, error):
        if self.enabled:
            self.records[object_id] = {
                "experiment_id": experiment_id,
                "reason": str(error),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            self.save_json(self.path, self.records)
