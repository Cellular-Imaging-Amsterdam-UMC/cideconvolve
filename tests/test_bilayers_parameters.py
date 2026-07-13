from bilayers_cli import BilayersJob, resolve_workflow_parameters


def test_parameters_json_overrides_bilayers_defaults() -> None:
    job = BilayersJob.from_cli([
        "--parameters",
        '{"iterations":"7","method":"ci_rl_tv","benchmark":true}',
    ])

    params = resolve_workflow_parameters(job.parameters)

    assert params.niter_list == [7]
    assert params.method == "ci_rl_tv"
    assert params.benchmark is True


def test_explicit_cli_overrides_parameters_json() -> None:
    job = BilayersJob.from_cli([
        "--parameters",
        '{"iterations":"7","method":"ci_rl_tv","benchmark":true}',
        "--method",
        "ci_rl",
    ])

    params = resolve_workflow_parameters(job.parameters)

    assert params.niter_list == [7]
    assert params.method == "ci_rl"
    assert params.benchmark is True


def test_time_range_parameters_are_resolved() -> None:
    job = BilayersJob.from_cli([
        "--parameters",
        '{"t_start":2,"t_stop":10,"t_step":3}',
    ])

    params = resolve_workflow_parameters(job.parameters)

    assert params.t_start == 2
    assert params.t_stop == 10
    assert params.t_step == 3


def test_output_dtype_parameter_is_resolved() -> None:
    job = BilayersJob.from_cli([
        "--parameters",
        '{"output_dtype":"uint16"}',
    ])

    params = resolve_workflow_parameters(job.parameters)

    assert params.output_dtype == "uint16"


def test_snr_mode_uses_value_only_for_manual() -> None:
    manual = resolve_workflow_parameters(BilayersJob.from_cli([
        "--parameters", '{"snr_mode":"manual","snr_value":4.2}',
    ]).parameters)
    automatic = resolve_workflow_parameters(BilayersJob.from_cli([
        "--parameters", '{"snr_mode":"auto","snr_value":99}',
    ]).parameters)
    disabled = resolve_workflow_parameters(BilayersJob.from_cli([
        "--parameters", '{"snr_mode":"none","snr_value":99}',
    ]).parameters)

    assert manual.snr == 4.2
    assert automatic.snr == "auto"
    assert disabled.snr is None


def test_legacy_snr_parameter_remains_supported() -> None:
    params = resolve_workflow_parameters(BilayersJob.from_cli([
        "--parameters", '{"snr":6.5}',
    ]).parameters)

    assert params.snr == 6.5
