from scripts import diffwave_train


def test_main_calls_train_diffwave(monkeypatch):
    called = {}

    def fake_train_diffwave():
        called["train"] = True

    monkeypatch.setattr(diffwave_train, "train_diffwave", fake_train_diffwave)
    diffwave_train.main()
    assert called.get("train"), "train_diffwave was not called by main()"


def test_script_runs_without_error(monkeypatch):
    monkeypatch.setattr(diffwave_train, "train_diffwave", lambda: None)
    diffwave_train.main()  # Should not raise
