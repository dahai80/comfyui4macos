import json
import os

import pytest

from custom_nodes4macos.pipeline.engine import PipelineEngine


def _write_checkpoint(job_dir, job_id, content_type, completed_stages,
                      completed_episodes=None, episode_finals=None,
                      scenes=None, story_title="测试剧"):
    os.makedirs(job_dir, exist_ok=True)
    cp = {
        "job_id": job_id,
        "content_type": content_type,
        "template_name": "series_drama" if content_type == "series" else "horror_short_drama",
        "completed_stages": completed_stages,
        "scenes": scenes or [],
        "artifacts": {},
        "config_overrides": {"story_title": story_title, "episode_count": len(scenes or [])},
        "character_registry": [],
        "global_style": "",
        "completed_episodes": completed_episodes or [],
        "episode_finals": episode_finals or [],
        "created_at": "2026-07-06T10:00:00",
        "updated_at": "2026-07-06T11:00:00",
    }
    with open(os.path.join(job_dir, "_checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False)
    return cp


def _touch(path, size=128):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


class TestEngineListJobs:
    def test_series_done_status_and_progress(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_abc"
        ep1 = str(job_dir / "测试剧_第一集.mp4")
        ep2 = str(job_dir / "测试剧_第二集.mp4")
        _write_checkpoint(
            str(job_dir), "20260706_abc", "series",
            ["story_ingest", "series_orchestrate"],
            completed_episodes=[1, 2],
            episode_finals=[ep1, ep2],
            scenes=[{"title": "第1集"}, {"title": "第2集"}],
        )
        eng = PipelineEngine(output_root=str(root))
        jobs = eng.list_jobs()
        assert len(jobs) == 1
        j = jobs[0]
        assert j["job_id"] == "20260706_abc"
        assert j["content_type"] == "series"
        assert j["status"] == "done"
        assert j["completed_episodes"] == [1, 2]
        assert j["total_episodes"] == 2
        assert j["episode_final_count"] == 2
        assert j["progress_label"] == "2/2 集"
        assert j["story_title"] == "测试剧"

    def test_series_in_progress(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_mid"
        _write_checkpoint(
            str(job_dir), "20260706_mid", "series",
            ["story_ingest"],
            completed_episodes=[1],
            episode_finals=[str(job_dir / "ep1.mp4")],
            scenes=[{"title": "第1集"}, {"title": "第2集"}, {"title": "第3集"}],
        )
        eng = PipelineEngine(output_root=str(root))
        j = eng.list_jobs()[0]
        assert j["status"] == "in_progress"
        assert j["progress_label"] == "1/3 集"

    def test_short_drama_done(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_short"
        _write_checkpoint(
            str(job_dir), "20260706_short", "short_drama",
            ["prompt_expand", "image_generate", "tts_synthesize", "ken_burns", "assemble", "sfx", "subtitle"],
        )
        eng = PipelineEngine(output_root=str(root))
        j = eng.list_jobs()[0]
        assert j["status"] == "done"
        assert j["total_episodes"] == 0
        assert "stages" in j["progress_label"]

    def test_jobs_sorted_newest_first(self, tmp_path):
        root = tmp_path / "out"
        _write_checkpoint(str(root / "job_a"), "job_a", "series", ["story_ingest"],
                          scenes=[{"title": "1"}], story_title="A")
        _write_checkpoint(str(root / "job_b"), "job_b", "series", ["story_ingest"],
                          scenes=[{"title": "1"}], story_title="B")
        eng = PipelineEngine(output_root=str(root))
        jobs = eng.list_jobs()
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == "job_b"

    def test_empty_root(self, tmp_path):
        eng = PipelineEngine(output_root=str(tmp_path / "empty"))
        assert eng.list_jobs() == []


class TestEngineGetJob:
    def test_detail_with_episode_finals(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_det"
        ep1 = str(job_dir / "测试剧_第一集.mp4")
        ep2 = str(job_dir / "测试剧_第二集.mp4")
        _write_checkpoint(
            str(job_dir), "20260706_det", "series",
            ["story_ingest", "series_orchestrate"],
            completed_episodes=[1, 2],
            episode_finals=[ep1, ep2],
            scenes=[{"title": "第1集"}, {"title": "第2集"}],
        )
        _touch(ep1, size=2048)
        _touch(ep2, size=1024)
        eng = PipelineEngine(output_root=str(root))
        job = eng.get_job("20260706_det")
        assert job is not None
        assert job["status"] == "done"
        finals = job["episode_finals"]
        assert len(finals) == 2
        assert finals[0]["episode"] == 1
        assert finals[0]["basename"] == "测试剧_第一集.mp4"
        assert finals[0]["exists"] is True
        assert finals[0]["size_bytes"] == 2048
        assert finals[1]["size_bytes"] == 1024

    def test_missing_job_returns_none(self, tmp_path):
        eng = PipelineEngine(output_root=str(tmp_path))
        assert eng.get_job("nope") is None

    def test_episode_finals_missing_file_marked(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_gone"
        ep1 = str(job_dir / "gone_第一集.mp4")
        _write_checkpoint(
            str(job_dir), "20260706_gone", "series",
            ["story_ingest"],
            completed_episodes=[1],
            episode_finals=[ep1],
            scenes=[{"title": "第1集"}],
        )
        eng = PipelineEngine(output_root=str(root))
        job = eng.get_job("20260706_gone")
        finals = job["episode_finals"]
        assert finals[0]["exists"] is False
        assert finals[0]["size_bytes"] == 0


class TestResolveJobFile:
    def test_resolves_existing_file(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_f"
        os.makedirs(job_dir)
        _touch(str(job_dir / "clip.mp4"))
        eng = PipelineEngine(output_root=str(root))
        p = eng.resolve_job_file("20260706_f", "clip.mp4")
        assert p is not None
        assert p.name == "clip.mp4"

    def test_rejects_path_traversal(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_f"
        os.makedirs(job_dir)
        eng = PipelineEngine(output_root=str(root))
        assert eng.resolve_job_file("20260706_f", "../../etc/passwd") is None
        assert eng.resolve_job_file("20260706_f", "../other_job/secret.mp4") is None

    def test_missing_job(self, tmp_path):
        eng = PipelineEngine(output_root=str(tmp_path))
        assert eng.resolve_job_file("nope", "x.mp4") is None

    def test_missing_file(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_f"
        os.makedirs(job_dir)
        eng = PipelineEngine(output_root=str(root))
        assert eng.resolve_job_file("20260706_f", "absent.mp4") is None

    def test_resolves_subpath_file(self, tmp_path):
        root = tmp_path / "out"
        job_dir = root / "20260706_sub"
        _touch(str(job_dir / "episode_02" / "scene_001_image.png"))
        eng = PipelineEngine(output_root=str(root))
        p = eng.resolve_job_file("20260706_sub", "episode_02/scene_001_image.png")
        assert p is not None
        assert p.suffix == ".png"


class TestServerRoutes:
    def test_handlers_use_env_output_root(self, tmp_path, monkeypatch):
        root = tmp_path / "out"
        job_dir = root / "20260706_env"
        ep1 = str(job_dir / "env_第一集.mp4")
        _write_checkpoint(
            str(job_dir), "20260706_env", "series",
            ["story_ingest", "series_orchestrate"],
            completed_episodes=[1],
            episode_finals=[ep1],
            scenes=[{"title": "第1集"}],
        )
        _touch(ep1)
        monkeypatch.setenv("DREAM_FACTORY_OUTPUT_ROOT", str(root))
        from custom_nodes4macos import server_routes

        jobs = server_routes.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "20260706_env"

        job = server_routes.get_job("20260706_env")
        assert job is not None
        assert job["episode_finals"][0]["exists"] is True

        p = server_routes.resolve_job_file("20260706_env", "env_第一集.mp4")
        assert p is not None

    def test_get_job_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DREAM_FACTORY_OUTPUT_ROOT", str(tmp_path))
        from custom_nodes4macos import server_routes

        assert server_routes.get_job("nope") is None

    def test_register_routes_noop_without_comfyui(self, monkeypatch):
        from custom_nodes4macos import server_routes

        monkeypatch.setattr(server_routes, "_COMFYUI_AVAILABLE", False)
        monkeypatch.setattr(server_routes, "_REGISTERED", False)
        server_routes.register_routes()
        assert server_routes._REGISTERED is False
