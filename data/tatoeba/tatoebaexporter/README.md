# Tatoeba Exporter (East Frisian)

Semi-automatic insertion of East Frisian translations into Tatoeba.

## Start

```
python tatoebaexporter.py
```

Requires only Python 3 with tkinter (standard library, no extra packages).

## Which sentences?

Only sentences that are verified in the verifier and not flagged for correction are offered. The selection matches `create_verified_submission.py` exactly (`verifyer/verified_indices.txt` minus `verifyer/sentences-to-correct.txt`).

## Workflow per sentence

1. The app checks via the Tatoeba API whether the sentence already has an East Frisian translation. The next 5 open sentences are checked in the background (one request every two seconds). The sentence currently shown is always checked individually first, so you can start right away. If it already has a translation, it is skipped as “already present”.
   If Tatoeba throttles requests (HTTP 429), the advance check stops. Individual checks then pause for 30 minutes.
2. **Copy & open Tatoeba**: The East Frisian sentence is copied to the clipboard and the Tatoeba page opens in the browser.
3. On Tatoeba, add the translation, choose the language “Eastern Frisian (frs)”, paste it and save.
4. **Inserted – verify & next**: The app queries the API again. If the translation is there, the sentence is marked as “inserted”. If not, you can still mark it as done or leave it open.
5. **Skip**: The sentence is skipped.

Progress is stored in `exporter_state.json` and persists between sessions.

## Why not fully automatic?

The Tatoeba API (api.tatoeba.org) is read-only. Writing requires logging in on the website. That is why the last step is done by hand in the browser.
