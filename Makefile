# Bear ColibrIA release verification. Python 3.12 on Linux. No network.
PY ?= python3

.PHONY: verify help
help:
	@echo "make verify — checksums + manifest + extract + stdlib-only import + service smoke"

verify:
	sha256sum -c SHA256SUMS
	rm -rf build/verify && mkdir -p build/verify
	cd build/verify && unzip -q ../../dist/exceedance_product_v1.zip
	cd build/verify && $(PY) -c "import hashlib, json, zipfile; z = zipfile.ZipFile('../../dist/exceedance_product_v1.zip'); m = json.loads(z.read('MANIFEST.json')); assert set(z.namelist()) == set(m['files']) | {'MANIFEST.json'}; [hashlib.sha256(z.read(n)).hexdigest() == h or (_ for _ in ()).throw(SystemExit('manifest mismatch: ' + n)) for n, h in m['files'].items()]; print('manifest: OK')"
	$(PY) -c "import json, zipfile; z = zipfile.ZipFile('dist/exceedance_product_v1.zip'); m = json.loads(z.read('MANIFEST.json')); [open(n,'rb').read() == z.read(n) or (_ for _ in ()).throw(SystemExit('tree differs from zip: ' + n)) for n in m['files']]; print('tree == zip: OK')"
	cd build/verify && $(PY) -B -c "import sys; import scripts.exceedance_service; assert not {'numpy','torch','ncps'} & set(sys.modules); print('stdlib-only: OK')"
	cd build/verify && $(PY) -B scripts/exceedance_service.py --ledger ledger.jsonl
	@echo "verify: OK"
