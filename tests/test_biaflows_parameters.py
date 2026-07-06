from biaflows_cli import BiaflowsJob, resolve_workflow_parameters


def test_parameters_json_overrides_descriptor_defaults() -> None:
    job = BiaflowsJob.from_cli([
        "--parameters",
        '{"iterations":"7","method":"ci_rl_tv","benchmark":true}',
    ])

    params = resolve_workflow_parameters(job.parameters)

    assert params.niter_list == [7]
    assert params.method == "ci_rl_tv"
    assert params.benchmark is True


def test_explicit_cli_overrides_parameters_json() -> None:
    job = BiaflowsJob.from_cli([
        "--parameters",
        '{"iterations":"7","method":"ci_rl_tv","benchmark":true}',
        "--method",
        "ci_rl",
    ])

    params = resolve_workflow_parameters(job.parameters)

    assert params.niter_list == [7]
    assert params.method == "ci_rl"
    assert params.benchmark is True
