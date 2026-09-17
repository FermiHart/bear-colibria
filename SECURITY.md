# SECURITY

This is a research artifact, not a production system. The threat model is
narrow and stated: checksums detect corruption, not malicious rollback; the
ledger proves pre-event issuance to a cooperative reader, not to an adversary
controlling the storage.

- Complete corrupt records fail closed. A same-implementation replay verifies
  determinism, not independent mathematical correctness.
- Report vulnerabilities privately to <contact@fermihart.com>. Do not open
  public issues with exploit details, meter data, or private paths.
