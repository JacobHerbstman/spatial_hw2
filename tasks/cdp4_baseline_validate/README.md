# cdp4_baseline_validate

Validates Julia baseline outputs against MATLAB reference `Hvectnoshock.mat` and runs deterministic rerun checks.

Output:
- `validation_report_4sector.csv`
- `validation_report_4sector_fast.csv`
- `validation_report_4sector_reference.csv`
- `parity_by_time_4sector.csv`
- `parity_by_time_4sector_fast.csv`
- `parity_by_time_4sector_reference.csv`

Validation mode labels:
- `smoke`
- `replication_reference`
- `replication_fast`

Make targets:
- `make -C code smoke`
- `make -C code replication_reference`
- `make -C code replication_fast`
