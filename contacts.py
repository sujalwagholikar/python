"""
contacts.py
===========
Resolves a contact name (e.g. "mom", "john") typed in a command into an
actual phone number, so you can say:

    "call mom"
    "open whatsapp and send hi to john"

Two sources are supported, checked in this order:
  1. A local contacts.json file you maintain yourself (fast, offline,
     100% reliable, recommended).
  2. If not found there and the input already looks like a phone number,
     it's used as-is.

Reading the phone's actual Contacts app content via ADB without root is
unreliable/inconsistent across OEMs and Android versions (Contacts
Provider queries via `content query` require permissions ADB shell
doesn't always have on stock ROMs), so this project intentionally uses
a simple local JSON file instead — it's the option that will actually
work reliably for you every time.

Edit contacts.json (auto-created on first run) like:
{
  "mom": "919876543210",
  "dad": "919812345678",
  "john": "14155552671"
}

Numbers should be in international format with country code, no +,
no spaces/dashes (this is what WhatsApp's wa.me links and Android
tel: intents expect).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CONTACTS_FILE = Path(__file__).parent / "contacts.json"


def _load_contacts() -> dict:
    if not CONTACTS_FILE.exists():
        CONTACTS_FILE.write_text(json.dumps({
            "example_mom": "919876543210",
            "example_dad": "919812345678"
        }, indent=2))
        return {}
    try:
        return json.loads(CONTACTS_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def _looks_like_number(s: str) -> bool:
    digits = re.sub(r"[^\d]", "", s)
    return len(digits) >= 7


def resolve_contact(name_or_number: str) -> str:
    """
    Resolve a name to a phone number using contacts.json, or pass through
    a value that already looks like a phone number.
    """
    cleaned = name_or_number.strip()

    if _looks_like_number(cleaned):
        # normalize: strip spaces/dashes/plus
        return re.sub(r"[^\d]", "", cleaned)

    contacts = _load_contacts()
    key = cleaned.lower()
    if key in contacts:
        return contacts[key]

    # fuzzy: try partial match
    for stored_name, number in contacts.items():
        if key in stored_name.lower() or stored_name.lower() in key:
            return number

    raise ValueError(
        f"Unknown contact '{name_or_number}'. Add them to "
        f"{CONTACTS_FILE.name} (in this project folder) like:\n"
        f'  "{key}": "91XXXXXXXXXX"\n'
        f"or use their phone number directly in the command."
    )


def add_contact(name: str, number: str) -> None:
    contacts = _load_contacts()
    contacts[name.lower().strip()] = re.sub(r"[^\d]", "", number)
    CONTACTS_FILE.write_text(json.dumps(contacts, indent=2))
