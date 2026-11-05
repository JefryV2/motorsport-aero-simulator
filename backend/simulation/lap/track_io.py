"""Parse a user-supplied centreline file.

Only coordinates in metres are accepted. There is no built-in circuit, no
geographic projection and no silent replacement of bad input: malformed data
raises :class:`TrackError` so the caller can show a real error.
"""

from __future__ import annotations

from backend.simulation.lap.track import TrackError


def _split(line: str) -> list[str]:
    cleaned = line.replace(",", " ").replace(";", " ").replace("\t", " ")
    return [token for token in cleaned.split(" ") if token]


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_centerline_csv(text: str) -> list[list[float]]:
    """Return ``[[x, y], ...]`` from CSV, TSV or whitespace-separated text.

    The first two numeric columns are x and y in metres; an optional third is
    kept as elevation. Header rows, unit rows and ``#``/``//`` comments are
    ignored. A row whose first two fields are not both numbers is skipped, so
    an ``id,x,y`` header does not become a data point.
    """
    if not isinstance(text, str) or not text.strip():
        raise TrackError("Centreline data is empty")

    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        tokens = _split(line)
        if len(tokens) < 2 or not (_is_number(tokens[0]) and _is_number(tokens[1])):
            continue
        values = [float(tokens[0]), float(tokens[1])]
        if len(tokens) >= 3 and _is_number(tokens[2]):
            values.append(float(tokens[2]))
        rows.append(values)

    if len(rows) < 8:
        raise TrackError(
            f"A centreline needs at least 8 coordinate rows, found {len(rows)}"
        )
    return rows
