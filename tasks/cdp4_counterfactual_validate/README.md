# cdp4_counterfactual_validate

Validates CDP-style 4-sector counterfactual outputs.

Outputs:
- `validation_counterfactual_4sector*.csv`
- `parity_by_time_counterfactual_4sector*.csv`
- `benchmark_validate_counterfactual_4sector*.csv`
- `dynamics_counterfactual_<profile>_<shock>.pdf`
- `dynamics_outer_counterfactual_<profile>_<shock>.pdf`
- `dynamics_parity_counterfactual_<profile>_<shock>.pdf`
- `table_counterfactual_summary_<profile>_<shock>.tex`
- `table_counterfactual_validation_<profile>_<shock>.tex`
- `table_counterfactual_parity_selected_t_<profile>_<shock>.tex`
- `key_econ_impacts_<profile>_<shock>.pdf`
- `key_econ_impacts_early_<profile>_<shock>.pdf`
- `table_key_econ_selected_t_<profile>_<shock>.tex`
- `table_key_econ_window_means_<profile>_<shock>.tex`

Validation modes:
- `identity_reference`
- `identity_fast`
- `toy_reference`
- `toy_fast`

Make targets:
- `make -C code smoke`
- `make -C code identity_reference`
- `make -C code identity_fast`
- `make -C code toy_reference`
- `make -C code toy_fast`
- `make -C code report_identity_fast`
- `make -C code report_identity_reference`
- `make -C code report_toy_fast`
- `make -C code report_toy_reference`
- `make -C code impact_report_toy_smoke_fast`
