# RELEASING — the anti-incident checklist

A previous deposit was withdrawn for a mixed-language document and improper
files. No release ships until every item below is checked:

1. **Single language** across the release (this repo: English).
2. **Allowlist only**: every shipped file is named in `ARTIFACTS.manifest`
   with its SHA256. No logs, keys, datasets, credentials, or personal paths.
3. **Review the package** file by file before tagging. Never edit a published
   release; ship a new version.
4. **Evidence frozen**: plan, one data attempt, replay-exact audit per claim.
