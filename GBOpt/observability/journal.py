# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Write and read a durable, opt-in JSONL event journal plus its companion run manifest.

This module consumes one already-built ``OptimizationEvent``/``RunManifest`` (from
``GBOpt.observability.types``) per call and serializes it to a UTF-8 file on disk, or
reads such a file back as plain, JSON-decoded data. It does not decide when a lifecycle
occurrence happens, what an event's fields mean, or run any optimizer logic; those
belong to ``GBOpt.observability.types`` and to ``MonteCarloMinimizer``/
``GeneticAlgorithmMinimizer``.

**Journal vs. checkpoint**: a journal file (or a manifest file) must never be read back
as optimizer restart state, and a checkpoint file must never be used as an event log.
The two serve different purposes and have different guarantees -- a checkpoint is a
compact, overwritten, authoritative snapshot of live optimizer state; a journal is an
append-only, durable record of what already happened, kept even after the run that
produced it has ended. Nothing in this module reconstructs an optimizer, a manipulator,
or a structure from journal/manifest data, and no reader here returns anything but
plain ``dict``/``list``/scalar JSON data -- never a domain object.

**Single-writer ownership**: exactly one ``JsonlEventSink`` instance, owned by the
optimization driver process, is expected to write to a given journal path for a given
run. This module does not implement cross-process locking -- a scheduler that runs
several evaluator workers must have each worker report back to the driver (e.g. via its
existing evaluator callback return value), never open the same journal path directly
from more than one process.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Self

from GBOpt.observability.types import (
    ObservabilityTypeError,
    ObservabilityValueError,
    OptimizationEvent,
    RunContext,
    RunManifest,
)

logger = logging.getLogger(__name__)

JournalWriteMode = Literal["append", "overwrite"]
_JOURNAL_WRITE_MODES: tuple[JournalWriteMode, ...] = ("append", "overwrite")


def _run_to_json_dict(run: RunContext) -> dict[str, object]:
    """Serialize one ``RunContext`` to a JSON-safe mapping.

    :param run: ``RunContext`` to serialize.
    :return: Plain ``dict`` with every field JSON-safe.
    """
    return {
        "run_id": run.run_id,
        "seed": run.seed,
        "algorithm": run.algorithm.value,
        "case_id": run.case_id,
        "campaign_id": run.campaign_id,
    }


def _event_to_json_dict(event: OptimizationEvent) -> dict[str, object]:
    """Serialize one ``OptimizationEvent`` to a JSON-safe mapping.

    Every field is written explicitly, translating each enum to its stable ``.value``
    and each read-only mapping to a plain ``dict`` -- ``json.dumps`` cannot serialize
    either directly.

    :param event: Event to serialize.
    :return: Plain ``dict`` with every field JSON-safe.
    """
    return {
        "schema_version": event.schema_version,
        "event_type": event.event_type.value,
        "run": _run_to_json_dict(event.run),
        "iteration": event.iteration,
        "candidate_id": event.candidate_id,
        "input_index": event.input_index,
        "status": None if event.status is None else event.status.value,
        "selection_energy": event.selection_energy,
        "energy": event.energy,
        "failure_stage": (
            None if event.failure_stage is None else event.failure_stage.value
        ),
        "failure_code": event.failure_code,
        "failure_message": event.failure_message,
        "operation_name": event.operation_name,
        "operation_parameters": (
            None
            if event.operation_parameters is None
            else dict(event.operation_parameters)
        ),
        "termination_reason": (
            None
            if event.termination_reason is None
            else event.termination_reason.value
        ),
    }


def _manifest_to_json_dict(manifest: RunManifest) -> dict[str, object]:
    """Serialize one ``RunManifest`` to a JSON-safe mapping.

    :param manifest: Manifest to serialize.
    :return: Plain ``dict`` with every field JSON-safe.
    """
    return {
        "schema_version": manifest.schema_version,
        "run": _run_to_json_dict(manifest.run),
        "created_at": manifest.created_at,
    }


class JsonlEventSink:
    """Append ``OptimizationEvent``\\ s to a durable, UTF-8, one-object-per-line file.

    **Encoding and line shape**: every line is exactly one JSON object followed by a
    single ``"\\n"`` (never ``"\\r\\n"``, regardless of platform), UTF-8 encoded. This
    is fixed, not configurable, so a journal file is portable across the platforms this
    project supports.

    **Append/overwrite policy**: ``mode="append"`` (the default) opens the file for
    appending, creating it if absent, and never truncates existing lines -- resuming a
    run (or pointing a second run's sink at the same path) preserves prior provenance.
    ``mode="overwrite"`` truncates the file first. Choosing ``"overwrite"`` for a path
    that already holds a different run's events destroys that run's journal; this is
    deliberate (the caller asked for it), not guarded against here.

    **Flush policy**: every ``emit()`` call writes its line and then calls
    ``flush()`` on the underlying file object before returning, so a reader opening the
    file after ``emit()`` returns sees every line written so far. This does not call
    ``os.fsync()`` -- a flushed line survives this process's own crash (the data has
    left Python's buffers) but is not guaranteed to survive an OS crash or power loss
    before the kernel writes it back to disk.

    **Write-failure policy**: an ``OSError`` raised by the underlying write or flush
    (e.g. disk full, permission denied, path removed) propagates to the caller
    unchanged -- this class never catches or retries it internally, since silently
    swallowing a durability failure would defeat the sink's entire purpose. What keeps
    a journal failure from ever reaching, or corrupting, the optimizer's own numerical
    state is the recovery boundary one layer up: ``CompositeEventSink`` isolates one
    failing sink from its siblings, and both ``MonteCarloMinimizer._emit``/
    ``GeneticAlgorithmMinimizer._emit`` already catch any sink exception and continue
    the run regardless. A caller using this sink directly, with no such boundary above
    it, is responsible for handling that propagated exception itself.

    **Single-writer ownership**: see this module's own docstring.

    :param path: Journal file path.
    :param mode: Keyword argument, optional, defaults to ``"append"``. ``"append"`` or
        ``"overwrite"``.
    :raises ObservabilityValueError: If ``mode`` is neither ``"append"`` nor
        ``"overwrite"``.
    :raises OSError: If the file cannot be opened (missing parent directory,
        permissions, ...).
    """

    def __init__(self, path: str | Path, *, mode: JournalWriteMode = "append") -> None:
        """Open one durable JSONL event journal for writing.

        :param path: Keyword argument omitted; journal file path.
        :param mode: Keyword argument, optional, defaults to ``"append"``.
            ``"append"`` or ``"overwrite"``.
        :raises ObservabilityValueError: If ``mode`` is neither ``"append"`` nor
            ``"overwrite"``.
        :raises OSError: If the file cannot be opened.
        """
        if mode not in _JOURNAL_WRITE_MODES:
            raise ObservabilityValueError(
                f"mode must be one of {_JOURNAL_WRITE_MODES!r}"
            )
        self._path = Path(path)
        file_mode = "a" if mode == "append" else "w"
        self._file = self._path.open(
            file_mode, encoding="utf-8", newline="\n"
        )
        self._closed = False

    @property
    def path(self) -> Path:
        """Journal file path this sink writes to."""
        return self._path

    def emit(self, event: OptimizationEvent) -> None:
        """Append one event as a single JSON line, then flush.

        :param event: Event to append.
        :raises ObservabilityTypeError: If ``event`` is not an ``OptimizationEvent``.
        :raises OSError: If the write or flush fails.
        :raises ValueError: If this sink is already closed (the underlying file
            object's own "I/O operation on closed file" error).
        """
        if not isinstance(event, OptimizationEvent):
            raise ObservabilityTypeError("event must be an OptimizationEvent")
        line = json.dumps(_event_to_json_dict(event), sort_keys=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file. Safe to call more than once."""
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> Self:
        """Return this sink unchanged, for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the underlying file on context-manager exit."""
        self.close()


def write_run_manifest(
    path: str | Path, manifest: RunManifest, *, overwrite: bool = False
) -> None:
    """Write one ``RunManifest`` as a single UTF-8 JSON object.

    **Overwrite policy**: a manifest is written once per run, so an existing file at
    ``path`` is left untouched and ``ObservabilityValueError`` is raised, unless the
    caller explicitly passes ``overwrite=True`` -- there is no implicit "append" for a
    manifest, since it is one object, not a line stream.

    :param path: Manifest file path.
    :param manifest: Keyword argument omitted; manifest to write.
    :param overwrite: Keyword argument, optional, defaults to ``False``. Replace an
        existing manifest file at ``path`` instead of raising.
    :raises ObservabilityTypeError: If ``manifest`` is not a ``RunManifest``.
    :raises ObservabilityValueError: If a manifest already exists at ``path`` and
        ``overwrite`` is not set.
    :raises OSError: If the file cannot be written.
    """
    if not isinstance(manifest, RunManifest):
        raise ObservabilityTypeError("manifest must be a RunManifest")
    manifest_path = Path(path)
    if manifest_path.exists() and not overwrite:
        raise ObservabilityValueError(
            f"a run manifest already exists at {manifest_path}; pass overwrite=True "
            "to replace it"
        )
    payload = _manifest_to_json_dict(manifest)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_file:
        json.dump(payload, manifest_file, sort_keys=True, indent=2)
        manifest_file.write("\n")
        manifest_file.flush()


def read_run_manifest(path: str | Path) -> dict[str, object]:
    """Read one manifest file back as plain, JSON-decoded data.

    This never reconstructs a ``RunManifest`` -- the return value is the raw JSON
    payload, for provenance inspection only.

    :param path: Manifest file path.
    :return: Decoded manifest object.
    :raises OSError: If the file cannot be read.
    :raises ObservabilityValueError: If the file's contents are not valid JSON.
    """
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ObservabilityValueError(
            f"run manifest {manifest_path} is not valid JSON: {exc}"
        ) from exc
    return decoded


def read_journal_events(path: str | Path) -> Iterator[dict[str, object]]:
    """Iterate one journal file's events as plain, JSON-decoded data, line by line.

    This never reconstructs an ``OptimizationEvent`` -- each yielded value is the raw
    JSON payload for one line, for provenance inspection only.

    **Truncated final line handling**: if the file's last non-empty line is not valid
    JSON (e.g. the writing process was killed mid-line before a subsequent ``flush()``
    completed), that final line is skipped and a warning is logged; every earlier,
    complete line is still yielded. A malformed line anywhere else in the file is a
    real corruption, not a truncation this format tolerates, and raises
    ``ObservabilityValueError`` immediately.

    :param path: Journal file path.
    :return: Decoded event objects, in file order.
    :raises OSError: If the file cannot be read.
    :raises ObservabilityValueError: If a non-final line is not valid JSON.
    """
    journal_path = Path(path)
    with journal_path.open("r", encoding="utf-8") as journal_file:
        lines = journal_file.readlines()
    last_content_index = max(
        (index for index, raw_line in enumerate(lines) if raw_line.strip()),
        default=-1,
    )
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError:
            if index == last_content_index:
                logger.warning(
                    "journal %s: truncated final line ignored", journal_path
                )
                return
            raise ObservabilityValueError(
                f"journal {journal_path} line {index + 1} is not valid JSON"
            ) from None


__all__ = [
    "JournalWriteMode",
    "JsonlEventSink",
    "write_run_manifest",
    "read_run_manifest",
    "read_journal_events",
]
