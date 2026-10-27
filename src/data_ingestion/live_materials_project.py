"""Bounded and resilient Materials Project API synchronizer."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class MaterialsProjectClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.materialsproject.org/materials/summary/",
        timeout: float = 20.0,
        supported_elements: set[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.supported_elements = supported_elements or set()

    def fetch_batch(self, skip: int, limit: int) -> list[dict[str, Any]]:
        """Fetch a bounded batch and apply supported-element filtering locally.

        Materials Project treats a comma-separated ``elements`` query as a
        combined constraint. Sending the whole supported-element registry
        therefore asks for materials containing every element and commonly
        returns an empty page.
        """
        page_skip = skip
        matched: list[dict[str, Any]] = []
        pages_checked = 0
        while len(matched) < limit and pages_checked < 20:
            records = self._fetch_page(page_skip, max(limit, 25))
            if not records:
                break
            if self.supported_elements:
                records = [
                    record for record in records
                    if set(record.get("elements", [])) <= self.supported_elements
                ]
            matched.extend(records)
            page_skip += max(limit, 25)
            pages_checked += 1
        return matched[:limit]

    def _fetch_page(self, skip: int, limit: int) -> list[dict[str, Any]]:
        fields = ",".join([
            "material_id",
            "formula_pretty",
            "elements",
            "structure",
            "formation_energy_per_atom",
        ])
        params = {"_skip": skip, "_limit": limit, "_fields": fields}
        query = urlencode(params)
        request = Request(
            f"{self.endpoint}?{query}",
            headers={
                "X-API-KEY": self.api_key,
                "Accept": "application/json",
                "User-Agent": "SYNTHESIS-ZERO/1.0",
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("results", []))
            return data if isinstance(data, list) else []
        return payload if isinstance(payload, list) else []


class LiveMaterialsProjectSync:
    """Poll Materials Project conservatively and fall back to local data."""

    def __init__(
        self,
        client: MaterialsProjectClient,
        ingest: Callable[[list[dict[str, Any]]], list[Any]],
        fallback_path: str,
        initial_batch: int = 10,
        min_batch: int = 5,
        max_batch: int = 25,
        interval_seconds: float = 300.0,
    ) -> None:
        self.client = client
        self.ingest = ingest
        self.fallback_path = fallback_path
        self.batch_size = max(min_batch, min(initial_batch, max_batch))
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.interval_seconds = interval_seconds
        self.offset = 0
        self.is_running = False
        self.thread: threading.Thread | None = None
        self.last_status = "not_started"
        self.last_error: str | None = None
        self.last_sync_time: str | None = None
        self.last_batch_count = 0
        self.total_remote_records = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_duplicates = 0
        self.fallback_batches = 0

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="materials-project-sync")
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)

    def sync_once(self) -> dict[str, Any]:
        records = None
        for attempt in range(3):
            try:
                records = self.client.fetch_batch(self.offset, self.batch_size)
                break
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                self.last_error = f"{type(error).__name__}: {error}"
                self.last_status = "retrying"
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
        try:
            if records is None:
                raise RuntimeError(self.last_error or "Materials Project request failed")
            self.last_error = None
            self.last_status = "remote_ok"
            self.total_remote_records += len(records)
            self.last_batch_count = len(records)
            if records:
                self.offset += len(records)
                results = self.ingest(records)
                self.total_accepted += sum(result.status == "accepted" for result in results)
                self.total_rejected += sum(result.status == "rejected" for result in results)
                self.total_duplicates += sum(result.status == "duplicate" for result in results)
                if len(records) == self.batch_size:
                    self.batch_size = min(self.max_batch, self.batch_size + 5)
            else:
                self.offset = 0
                self.batch_size = self.min_batch
                self.last_status = "remote_empty"
                self._load_fallback()
            self.last_sync_time = time.strftime("%Y-%m-%d %H:%M:%S")
            return self.status()
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self.last_status = "fallback"
            self._load_fallback()
            self.batch_size = max(self.min_batch, self.batch_size // 2)
            self.last_sync_time = time.strftime("%Y-%m-%d %H:%M:%S")
            return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "status": self.last_status,
            "running": self.is_running,
            "batch_size": self.batch_size,
            "offset": self.offset,
            "last_batch_count": self.last_batch_count,
            "last_sync_time": self.last_sync_time,
            "last_error": self.last_error,
            "total_remote_records": self.total_remote_records,
            "accepted": self.total_accepted,
            "rejected": self.total_rejected,
            "duplicates": self.total_duplicates,
            "fallback_batches": self.fallback_batches,
        }

    def _run(self) -> None:
        while self.is_running:
            self.sync_once()
            deadline = time.monotonic() + self.interval_seconds
            while self.is_running and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))

    def _load_fallback(self) -> None:
        if not os.path.exists(self.fallback_path):
            return
        try:
            with open(self.fallback_path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            if isinstance(records, list) and records:
                results = self.ingest(records[: self.batch_size])
                self.fallback_batches += 1
                self.total_accepted += sum(result.status == "accepted" for result in results)
                self.total_rejected += sum(result.status == "rejected" for result in results)
                self.total_duplicates += sum(result.status == "duplicate" for result in results)
        except (OSError, ValueError, TypeError) as error:
            self.last_error = f"fallback {type(error).__name__}: {error}"
