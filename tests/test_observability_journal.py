# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for the durable JSONL event journal and run manifest."""

import json

import pytest

from GBOpt.evaluation import EvaluationStatus
from GBOpt.observability.journal import (
    JsonlEventSink,
    read_journal_events,
    read_run_manifest,
    write_run_manifest,
)
from GBOpt.observability.types import (
    ObservabilityTypeError,
    ObservabilityValueError,
    OptimizationAlgorithm,
    OptimizationEvent,
    OptimizationEventType,
    RunContext,
    RunManifest,
    TerminationReason,
)


def _run(**overrides):
    arguments = {
        "run_id": "run-1",
        "seed": 42,
        "algorithm": OptimizationAlgorithm.MONTE_CARLO,
    }
    arguments.update(overrides)
    return RunContext(**arguments)


def _event(**overrides):
    arguments = {
        "event_type": OptimizationEventType.RUN_STARTED,
        "run": _run(),
        "iteration": 0,
    }
    arguments.update(overrides)
    return OptimizationEvent(**arguments)


# --- JsonlEventSink -----------------------------------------------------------------


def test_emit_writes_one_json_line_conforming_to_the_event_schema(tmp_path):
    path = tmp_path / "journal.jsonl"
    sink = JsonlEventSink(path)

    sink.emit(_event())
    sink.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    decoded = json.loads(lines[0])
    assert decoded["schema_version"] == 1
    assert decoded["event_type"] == "run_started"
    assert decoded["run"]["run_id"] == "run-1"
    assert decoded["run"]["algorithm"] == "monte_carlo"


def test_emit_uses_utf8_and_lf_only_line_endings(tmp_path):
    path = tmp_path / "journal.jsonl"
    sink = JsonlEventSink(path)

    sink.emit(_event())
    sink.close()

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    raw.decode("utf-8")  # must not raise


def test_emit_flushes_before_returning_so_a_concurrent_reader_sees_the_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    sink = JsonlEventSink(path)

    sink.emit(_event())

    # Sink is still open (not closed) -- flush(), not close(), is what makes this
    # visible to an independent reader of the same path.
    assert path.read_text(encoding="utf-8").strip() != ""
    sink.close()


def test_emit_appends_multiple_events_in_order(tmp_path):
    path = tmp_path / "journal.jsonl"
    sink = JsonlEventSink(path)

    sink.emit(_event(iteration=0))
    sink.emit(_event(iteration=1))
    sink.emit(_event(iteration=2))
    sink.close()

    decoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [entry["iteration"] for entry in decoded] == [0, 1, 2]


def test_emit_rejects_non_optimization_event(tmp_path):
    sink = JsonlEventSink(tmp_path / "journal.jsonl")

    with pytest.raises(ObservabilityTypeError):
        sink.emit("not-an-event")

    sink.close()


def test_default_mode_appends_to_an_existing_file(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"marker": "prior-run"}\n', encoding="utf-8")

    sink = JsonlEventSink(path)
    sink.emit(_event())
    sink.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"marker": "prior-run"}


def test_overwrite_mode_truncates_an_existing_file(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"marker": "prior-run"}\n', encoding="utf-8")

    sink = JsonlEventSink(path, mode="overwrite")
    sink.emit(_event())
    sink.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_type"] == "run_started"


def test_rejects_unknown_mode(tmp_path):
    with pytest.raises(ObservabilityValueError):
        JsonlEventSink(tmp_path / "journal.jsonl", mode="clobber")


def test_write_failure_propagates_rather_than_being_swallowed(tmp_path):
    path = tmp_path / "journal.jsonl"
    sink = JsonlEventSink(path)
    sink.close()  # closing first means the next write hits a closed file

    with pytest.raises(ValueError):
        sink.emit(_event())


def test_close_is_idempotent(tmp_path):
    sink = JsonlEventSink(tmp_path / "journal.jsonl")

    sink.close()
    sink.close()  # must not raise


def test_context_manager_closes_on_exit(tmp_path):
    path = tmp_path / "journal.jsonl"

    with JsonlEventSink(path) as sink:
        sink.emit(_event())

    with pytest.raises(ValueError):
        sink.emit(_event())


def test_operation_parameters_are_embedded_by_reference_only_no_atom_arrays(tmp_path):
    path = tmp_path / "journal.jsonl"
    event = _event(
        event_type=OptimizationEventType.PROPOSAL_EVALUATED,
        candidate_id="candidate-1",
        operation_name="insert_atoms",
        operation_parameters={"count": 3, "site": "interface"},
        status=EvaluationStatus.SUCCESS,
        energy=1.5,
        selection_energy=1.5,
    )

    with JsonlEventSink(path) as sink:
        sink.emit(event)

    decoded = json.loads(path.read_text(encoding="utf-8").strip())
    assert decoded["operation_parameters"] == {"count": 3, "site": "interface"}
    assert "atoms" not in decoded


# --- read_journal_events -------------------------------------------------------------


def test_read_journal_events_round_trips_every_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    with JsonlEventSink(path) as sink:
        sink.emit(_event(iteration=0))
        sink.emit(_event(iteration=1))

    events = list(read_journal_events(path))

    assert [event["iteration"] for event in events] == [0, 1]


def test_read_journal_events_skips_blank_lines(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps({"iteration": 0}) + "\n\n" + json.dumps({"iteration": 1}) + "\n",
        encoding="utf-8",
    )

    events = list(read_journal_events(path))

    assert [event["iteration"] for event in events] == [0, 1]


def test_read_journal_events_ignores_a_truncated_final_line(tmp_path, caplog):
    path = tmp_path / "journal.jsonl"
    complete_line = json.dumps({"iteration": 0})
    path.write_text(complete_line + "\n" + '{"iteration": 1, "trunc', encoding="utf-8")

    with caplog.at_level("WARNING", logger="GBOpt.observability.journal"):
        events = list(read_journal_events(path))

    assert [event["iteration"] for event in events] == [0]
    assert any("truncated" in record.message for record in caplog.records)


def test_read_journal_events_raises_on_a_malformed_non_final_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text(
        '{"iteration": 0, "trunc\n' + json.dumps({"iteration": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ObservabilityValueError):
        list(read_journal_events(path))


def test_read_journal_events_handles_an_empty_file(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text("", encoding="utf-8")

    assert list(read_journal_events(path)) == []


# --- run manifest --------------------------------------------------------------------


def test_write_and_read_run_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = RunManifest(
        run=_run(run_id="run-42", case_id="case-a", campaign_id="campaign-b"),
        created_at="2026-01-01T00:00:00+00:00",
    )

    write_run_manifest(path, manifest)
    decoded = read_run_manifest(path)

    assert decoded["run"]["run_id"] == "run-42"
    assert decoded["run"]["case_id"] == "case-a"
    assert decoded["run"]["campaign_id"] == "campaign-b"
    assert decoded["created_at"] == "2026-01-01T00:00:00+00:00"


def test_manifest_and_journal_share_the_same_run_id(tmp_path):
    run = _run(run_id="shared-run-id")
    manifest_path = tmp_path / "manifest.json"
    journal_path = tmp_path / "journal.jsonl"

    write_run_manifest(manifest_path, RunManifest(run=run))
    with JsonlEventSink(journal_path) as sink:
        sink.emit(
            OptimizationEvent(
                event_type=OptimizationEventType.RUN_TERMINATED,
                run=run,
                iteration=5,
                termination_reason=TerminationReason.MAX_STEPS,
            )
        )

    manifest_data = read_run_manifest(manifest_path)
    (event_data,) = list(read_journal_events(journal_path))
    assert manifest_data["run"]["run_id"] == event_data["run"]["run_id"] == "shared-run-id"


def test_write_run_manifest_refuses_to_overwrite_by_default(tmp_path):
    path = tmp_path / "manifest.json"
    write_run_manifest(path, RunManifest(run=_run()))

    with pytest.raises(ObservabilityValueError):
        write_run_manifest(path, RunManifest(run=_run(run_id="run-2")))


def test_write_run_manifest_overwrite_true_replaces_existing_file(tmp_path):
    path = tmp_path / "manifest.json"
    write_run_manifest(path, RunManifest(run=_run(run_id="run-1")))

    write_run_manifest(path, RunManifest(run=_run(run_id="run-2")), overwrite=True)

    decoded = read_run_manifest(path)
    assert decoded["run"]["run_id"] == "run-2"


def test_write_run_manifest_rejects_non_run_manifest(tmp_path):
    with pytest.raises(ObservabilityTypeError):
        write_run_manifest(tmp_path / "manifest.json", "not-a-manifest")


def test_read_run_manifest_raises_on_malformed_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ObservabilityValueError):
        read_run_manifest(path)
