"""Small read-only client for the Tatoeba API (api.tatoeba.org)."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.tatoeba.org/v1"
SITE_BASE = "https://tatoeba.org/eng/sentences/show"
TIMEOUT_SECONDS = 30
USER_AGENT = "tatoebaexporter/1.0"
COOLDOWN_SECONDS = 30 * 60

_blocked_until = 0.0


class TatoebaError(Exception):
    """Tatoeba is unreachable or returned an unexpected response."""


class TatoebaRateLimited(TatoebaError):
    """Tatoeba is throttling requests (HTTP 429)."""


def sentence_url(sentence_id):
    return f"{SITE_BASE}/{sentence_id}"


def get_direct_translations(sentence_id, lang):
    """Return the texts of the directly linked translations in `lang`."""
    global _blocked_until
    if time.time() < _blocked_until:
        raise TatoebaRateLimited("Tatoeba is throttling requests (cooldown active)")
    query = urllib.parse.urlencode({"showtrans:lang": lang})
    url = f"{API_BASE}/sentences/{sentence_id}?{query}"
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 429:
            _blocked_until = time.time() + COOLDOWN_SECONDS
            raise TatoebaRateLimited("Tatoeba is throttling requests") from error
        raise TatoebaError(f"HTTP {error.code} for sentence {sentence_id}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise TatoebaError(f"No connection to Tatoeba ({error})") from error
    except ValueError as error:
        raise TatoebaError("Invalid response from Tatoeba") from error

    translations = payload.get("data", {}).get("translations", [])
    return [
        item["text"]
        for item in translations
        if item.get("lang") == lang and item.get("is_direct")
    ]
