from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path


class GraphStage:
    """Prepare dataset/project artifacts and persist every intermediate step."""

    def __init__(self, dataset_path: Path, run_dir: Path):
        self.dataset_path = dataset_path
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, value) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_dataset(self) -> list[dict]:
        records = []
        if self.dataset_path.is_dir():
            files = self.dataset_path.rglob("*.json")
            for file_path in files:
                records.append(json.loads(file_path.read_text(encoding="utf-8")))
        else:
            with zipfile.ZipFile(self.dataset_path) as archive:
                for name in archive.namelist():
                    if name.endswith(".json") and not name.startswith("__MACOSX/"):
                        records.append(json.loads(archive.read(name)))
        self.save_json("01_dataset.json", records)
        return records

    def clone_projects(self, records: list[dict]) -> list[dict]:
        projects_dir = self.run_dir / "projects"
        projects_dir.mkdir(exist_ok=True)
        results = []
        seen = set()
        for record in records:
            project = record.get("project")
            if not project or project in seen:
                continue
            seen.add(project)
            target = projects_dir / project.replace("/", "__")
            url = project if project.startswith(("http://", "https://")) else f"https://github.com/{project}.git"
            item = {"project": project, "url": url, "path": str(target), "status": "skipped"}
            if target.exists():
                item["status"] = "exists"
            else:
                try:
                    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True,
                                   capture_output=True, text=True)
                    item["status"] = "cloned"
                except subprocess.CalledProcessError as error:
                    item["status"] = "error"
                    item["error"] = error.stderr[-2000:]
            results.append(item)
        self.save_json("02_projects.json", results)
        return results

    def prepare_records(self, records: list[dict]) -> list[dict]:
        prepared = []
        for record in records:
            prepared.append({
                "id": record.get("id"),
                "language": record.get("language"),
                "project": record.get("project"),
                "summary": record.get("summary"),
                "details": record.get("details"),
                "cwe_ids": record.get("cwe_ids", []),
                "before": record.get("before", {}),
                "after": record.get("after", {}),
                "updates": record.get("updates", []),
            })
        self.save_json("03_prepared_records.json", prepared)
        return prepared
