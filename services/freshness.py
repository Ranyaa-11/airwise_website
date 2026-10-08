"""Shared data-freshness thresholds and IST-aware age classification."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

RECENT_DAYS = 5
OUTDATED_DAYS = 14
IST = ZoneInfo("Asia/Kolkata")


def freshness(last_date):
    if last_date is None:
        return {"age_days": None, "status": "too_old"}

    if isinstance(last_date, str):
        measured_date = date.fromisoformat(last_date[:10])
    elif isinstance(last_date, datetime):
        measured_date = last_date.date()
    elif isinstance(last_date, date):
        measured_date = last_date
    else:
        raise TypeError("last_date must be a date, datetime, ISO date string, or None")

    age_days = (datetime.now(IST).date() - measured_date).days
    status = "recent" if age_days <= RECENT_DAYS else (
        "outdated" if age_days <= OUTDATED_DAYS else "too_old"
    )
    return {"age_days": age_days, "status": status}


def source_age_minutes(source_timestamp, now=None):
    """Return the age of a source timestamp, treating naive timestamps as IST."""
    if source_timestamp is None:
        return None

    if isinstance(source_timestamp, str):
        try:
            source_timestamp = datetime.fromisoformat(
                source_timestamp.replace("Z", "+00:00")
            )
        except ValueError:
            return None

    if not isinstance(source_timestamp, datetime):
        raise TypeError("source_timestamp must be a datetime, ISO timestamp string, or None")

    source_timestamp = (
        source_timestamp.replace(tzinfo=IST)
        if source_timestamp.tzinfo is None
        else source_timestamp.astimezone(IST)
    )
    current = now or datetime.now(IST)
    current = (
        current.replace(tzinfo=IST)
        if current.tzinfo is None
        else current.astimezone(IST)
    )
    age = (current - source_timestamp).total_seconds() / 60
    return round(max(0.0, age), 1)
