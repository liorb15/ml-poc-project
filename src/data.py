"""Student-owned dataset loading contract.

Current implementation uses the Mikrokosmos difficulty dataset as a working
prototype while waiting for broader piano datasets such as CIPI.
"""

from __future__ import annotations

import csv
import statistics
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR

MIKROKOSMOS_DIR = DATA_DIR / "external" / "Mikrokosmos-difficulty"
MIKROKOSMOS_XML_DIR = MIKROKOSMOS_DIR / "musicxml"
MIKROKOSMOS_METADATA_FILE = MIKROKOSMOS_DIR / "metadata" / "mikrokosmos_metadata.csv"
BOOK_CODES = {
    "Mikrokosmos, Volumes I-II": 1,
    "Mikrokosmos, Volumes III-IV": 2,
    "Mikrokosmos, Volumes V-VI": 3,
}


def _coarse_difficulty(label: str) -> str:
    level = int(label.split()[1])
    if level <= 3:
        return "beginner"
    if level <= 6:
        return "intermediate"
    return "advanced"


@lru_cache(maxsize=1)
def _load_mikrokosmos_dataframe() -> pd.DataFrame:
    if not MIKROKOSMOS_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Mikrokosmos metadata file not found: {MIKROKOSMOS_METADATA_FILE}"
        )

    rows: list[dict[str, Any]] = []
    with MIKROKOSMOS_METADATA_FILE.open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            if row["4 hands"] != "FALSE":
                continue

            piece_id = int(row["piece number"])
            xml_path = MIKROKOSMOS_XML_DIR / f"{piece_id}.xml"
            if not xml_path.exists():
                continue

            feature_row = _extract_symbolic_features(xml_path)
            feature_row["piece_id"] = piece_id
            feature_row["work"] = row["work"]
            feature_row["book"] = row["book"]
            feature_row["book_code"] = BOOK_CODES.get(row["book"], 0)
            feature_row["composer"] = row["composer"]
            feature_row["henle_difficulty"] = row["henle_difficulty"]
            feature_row["difficulty_label"] = _coarse_difficulty(row["henle_difficulty"])
            rows.append(feature_row)

    if not rows:
        raise ValueError("No Mikrokosmos rows were loaded from the local dataset.")

    return pd.DataFrame(rows).sort_values("piece_id").reset_index(drop=True)


def _extract_symbolic_features(xml_path: Path) -> dict[str, float]:
    root = ET.parse(xml_path).getroot()
    note_nodes = root.findall(".//note")
    measure_nodes = root.findall(".//measure")

    played_notes = 0
    rests = 0
    chord_notes = 0
    durations: list[int] = []
    pitch_values: list[int] = []
    pitch_intervals: list[int] = []
    previous_pitch: int | None = None
    tempo_values: list[float] = []
    key_signature_values: list[int] = []
    time_signatures: list[tuple[int, int]] = []

    midi_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    for sound in root.findall(".//sound"):
        tempo_text = sound.attrib.get("tempo")
        if tempo_text is None:
            continue
        try:
            tempo_values.append(float(tempo_text))
        except ValueError:
            continue

    for attributes in root.findall(".//attributes"):
        fifths_text = attributes.findtext("key/fifths")
        if fifths_text is not None:
            try:
                key_signature_values.append(abs(int(fifths_text)))
            except ValueError:
                pass

        beats_text = attributes.findtext("time/beats")
        beat_type_text = attributes.findtext("time/beat-type")
        if beats_text is not None and beat_type_text is not None:
            try:
                time_signatures.append((int(beats_text), int(beat_type_text)))
            except ValueError:
                pass

    for note in note_nodes:
        duration_text = note.findtext("duration")
        if duration_text and duration_text.isdigit():
            durations.append(int(duration_text))

        if note.find("rest") is not None:
            rests += 1
            continue

        played_notes += 1
        if note.find("chord") is not None:
            chord_notes += 1

        step = note.findtext("pitch/step")
        octave = note.findtext("pitch/octave")
        alter = note.findtext("pitch/alter")
        if step and octave:
            pitch = (int(octave) + 1) * 12 + midi_map[step] + int(alter or 0)
            pitch_values.append(pitch)
            if previous_pitch is not None:
                pitch_intervals.append(abs(pitch - previous_pitch))
            previous_pitch = pitch

    measure_count = len(measure_nodes)
    pitch_span = max(pitch_values) - min(pitch_values) if pitch_values else 0
    avg_duration = statistics.mean(durations) if durations else 0.0
    duration_std = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    avg_pitch_interval = statistics.mean(pitch_intervals) if pitch_intervals else 0.0
    max_pitch_interval = max(pitch_intervals) if pitch_intervals else 0.0
    unique_pitch_count = len(set(pitch_values)) if pitch_values else 0
    rest_ratio = float(rests / len(note_nodes)) if note_nodes else 0.0
    accidental_ratio = float(sum(1 for n in pitch_values if (n % 12) not in {0,2,4,5,7,9,11}) / played_notes) if played_notes else 0.0
    tempo_mean = statistics.mean(tempo_values) if tempo_values else 0.0
    key_signature_complexity = statistics.mean(key_signature_values) if key_signature_values else 0.0
    time_signature_changes = max(len(set(time_signatures)) - 1, 0) if time_signatures else 0.0
    notes_per_second_proxy = float(played_notes * tempo_mean / 60.0) if tempo_mean else 0.0

    return {
        "notes_total": float(len(note_nodes)),
        "notes_played": float(played_notes),
        "rests": float(rests),
        "measures": float(measure_count),
        "chord_notes": float(chord_notes),
        "pitch_span": float(pitch_span),
        "avg_duration": float(avg_duration),
        "notes_per_measure": float(played_notes / measure_count) if measure_count else 0.0,
        "chord_ratio": float(chord_notes / played_notes) if played_notes else 0.0,
        "rest_ratio": float(rest_ratio),
        "unique_pitch_count": float(unique_pitch_count),
        "avg_pitch_interval": float(avg_pitch_interval),
        "max_pitch_interval": float(max_pitch_interval),
        "tempo_mean": float(tempo_mean),
        "notes_per_second_proxy": float(notes_per_second_proxy),
        "duration_std": float(duration_std),
        "accidental_ratio": float(accidental_ratio),
        "key_signature_complexity": float(key_signature_complexity),
        "time_signature_changes": float(time_signature_changes),
    }


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    The current prototype uses symbolic features extracted from the local
    Mikrokosmos difficulty dataset and predicts a coarse difficulty label.
    """

    dataset_df = _load_mikrokosmos_dataframe()
    feature_columns = [
        "notes_total",
        "notes_played",
        "rests",
        "measures",
        "chord_notes",
        "pitch_span",
        "avg_duration",
        "notes_per_measure",
        "chord_ratio",
        "rest_ratio",
        "unique_pitch_count",
        "avg_pitch_interval",
        "max_pitch_interval",
        "tempo_mean",
        "notes_per_second_proxy",
        "duration_std",
        "accidental_ratio",
        "key_signature_complexity",
        "time_signature_changes",
        "book_code",
    ]

    X = dataset_df[feature_columns].copy()
    y = dataset_df["difficulty_label"].copy()

    return tuple(
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )
