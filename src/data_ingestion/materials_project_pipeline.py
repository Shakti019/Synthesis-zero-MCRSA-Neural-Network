"""Audited, validated, unique ingestion for Materials Project records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class IngestionResult:
    status: str
    fingerprint: str
    reason: str
    record: dict[str, Any] | None = None


class MaterialsProjectPipeline:
    """Track every record and admit only validated unique training samples."""

    def __init__(self, audit_path: str, accepted_path: str, supported_elements: set[str]) -> None:
        self.audit_path = audit_path
        self.accepted_path = accepted_path
        self.supported_elements = supported_elements
        self._accepted_fingerprints = self._load_fingerprints()
        self._lock = threading.Lock()

    def ingest(self, records: Iterable[dict[str, Any]]) -> list[IngestionResult]:
        results = []
        for record in records:
            results.append(self._ingest_one(record))
        return results

    def normalize_for_inference(
        self, records: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate and normalize records without writing audit or registry files."""
        normalized_records = []
        for record in records:
            normalized, reason = self._normalize_and_validate(record)
            if normalized is None:
                raise ValueError(f"record rejected for inference: {reason}")
            normalized_records.append(normalized)
        return normalized_records

    def accepted_records(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.accepted_path):
            return []
        with open(self.accepted_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if os.path.exists(self.audit_path):
            with open(self.audit_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        status = json.loads(line)["status"]
                    except (ValueError, KeyError):
                        continue
                    counts[status] = counts.get(status, 0) + 1
        return counts

    def _ingest_one(self, raw: dict[str, Any]) -> IngestionResult:
        with self._lock:
            if not isinstance(raw, dict):
                return self._reject("invalid_record", "record must be an object", raw)
            fingerprint = self._fingerprint(raw)
            if fingerprint in self._accepted_fingerprints:
                return self._record_result("duplicate", fingerprint, "record already ingested")

            normalized, reason = self._normalize_and_validate(raw)
            if normalized is None:
                return self._record_result("rejected", fingerprint, reason)

            self._accepted_fingerprints.add(fingerprint)
            accepted = self.accepted_records()
            accepted.append(normalized)
            self._write_json(self.accepted_path, accepted)
            return self._record_result("accepted", fingerprint, "validated unique training record", normalized)

    def _normalize_and_validate(self, raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        material_id = raw.get("material_id") or raw.get("material_ids")
        formula = raw.get("formula_pretty") or raw.get("formula") or raw.get("formula_reduced")
        elements = raw.get("elements")
        coordinates = raw.get("coordinates_3d")
        if coordinates is None:
            structure = raw.get("structure") or {}
            sites = structure.get("sites") if isinstance(structure, dict) else None
            if isinstance(sites, list):
                site_elements = []
                for site in sites:
                    species = site.get("species") if isinstance(site, dict) else None
                    if isinstance(species, list) and species:
                        element = species[0].get("element")
                        if isinstance(element, str):
                            site_elements.append(element)
                coordinates = [
                    site.get("xyz") for site in sites
                    if isinstance(site, dict)
                ]
                if len(site_elements) == len(coordinates):
                    elements = site_elements
        energy = raw.get("formation_energy_per_atom")
        if energy is None:
            energy = raw.get("target_energy", raw.get("formation_energy"))
        if isinstance(material_id, list):
            material_id = material_id[0] if material_id else None
        if not isinstance(formula, str) or not formula.strip():
            return None, "missing formula"
        if not isinstance(elements, list) or not elements:
            return None, "missing elements"
        if any(element not in self.supported_elements for element in elements):
            return None, "contains unsupported element"
        if not isinstance(coordinates, list) or len(coordinates) != len(elements):
            return None, "coordinates and elements must have equal nonzero length"
        if any(
            not isinstance(point, (list, tuple)) or len(point) != 3
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
            for point in coordinates
        ):
            return None, "coordinates must be finite 3D points"
        if not isinstance(energy, (int, float)) or not math.isfinite(energy):
            return None, "missing or non-finite formation energy"
        return {
            "material_id": material_id,
            "formula": formula.strip(),
            "elements": elements,
            "coordinates_3d": coordinates,
            "target_energy": float(energy),
            "source": "materials_project",
            "source_record_id": material_id,
        }, ""

    def _fingerprint(self, raw: dict[str, Any]) -> str:
        identity = raw.get("material_id") or raw.get("material_ids")
        if isinstance(identity, list):
            identity = identity[0] if identity else None
        if identity:
            value = f"materials_project:{identity}"
        else:
            value = json.dumps(
                {
                    "formula": raw.get("formula_pretty") or raw.get("formula"),
                    "elements": raw.get("elements"),
                    "coordinates_3d": raw.get("coordinates_3d"),
                    "energy": raw.get("formation_energy_per_atom", raw.get("target_energy")),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _reject(self, status: str, reason: str, raw: Any) -> IngestionResult:
        fingerprint = self._fingerprint(raw if isinstance(raw, dict) else {"raw": str(raw)})
        return self._record_result(status, fingerprint, reason)

    def _record_result(
        self,
        status: str,
        fingerprint: str,
        reason: str,
        record: dict[str, Any] | None = None,
    ) -> IngestionResult:
        audit = {
            "status": status,
            "fingerprint": fingerprint,
            "reason": reason,
            "record": record,
        }
        with open(self.audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit) + "\n")
        return IngestionResult(status, fingerprint, reason, record)

    def _load_fingerprints(self) -> set[str]:
        if not os.path.exists(self.audit_path):
            return set()
        fingerprints = set()
        with open(self.audit_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                    if entry.get("status") == "accepted":
                        fingerprints.add(entry["fingerprint"])
                except (ValueError, KeyError):
                    continue
        return fingerprints

    @staticmethod
    def _write_json(path: str, value: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
