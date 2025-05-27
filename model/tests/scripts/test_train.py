from scripts import train


def test_main_calls_train(monkeypatch):
    called = {}

    def fake_train():
        called["train"] = True

    monkeypatch.setattr(train, "train", fake_train)
    train.main()
    assert called.get("train"), "train() was not called by main()"


def test_script_runs_without_error(monkeypatch):
    monkeypatch.setattr(train, "train", lambda: None)
    train.main()  # Should not raise
