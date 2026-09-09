"""Parse reusable calendar-month manifests for RTMA datasets and utilities."""

import re
from pathlib import Path


MONTH_PATTERN = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


def parse_month(value, source="month selection"):
    """Parse one strict ``YYYY-MM`` value into an integer ``(year, month)`` key."""
    match = MONTH_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"{source}: invalid month {value!r}; expected exact YYYY-MM")
    return int(match.group(1)), int(match.group(2))


def read_month_manifest(path, option_name="month manifest"):
    """Read unique month keys, ignoring blank lines and full-line ``#`` comments."""
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"{option_name}: month manifest not found: {manifest}")

    months = []
    seen = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            key = parse_month(value, source=f"{option_name} ({manifest}:{line_number})")
            if key in seen:
                raise ValueError(
                    f"{option_name}: duplicate month {value!r} at "
                    f"{manifest}:{line_number}"
                )
            seen.add(key)
            months.append(key)

    if not months:
        raise ValueError(f"{option_name}: month manifest contains no months: {manifest}")
    return sorted(months)


def format_months(months):
    """Return canonical ``YYYY-MM`` strings for month keys."""
    return [f"{year:04d}-{month:02d}" for year, month in months]
