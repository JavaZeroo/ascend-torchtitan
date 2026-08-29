from ascend_titan.tools import doctor


def test_probe_runs_anywhere():
    r = doctor.probe(import_torchtitan=False)
    assert r.torch  # torch is a test dependency
    text = doctor.render(r)
    assert "ascend-titan doctor" in text


def test_cli_json(capsys):
    assert doctor.main(["--json", "--no-titan"]) == 0
    out = capsys.readouterr().out
    assert '"torch"' in out
