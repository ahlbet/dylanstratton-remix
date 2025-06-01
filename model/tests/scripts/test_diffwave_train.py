from scripts import diffwave_train
import sys


def test_main_calls_train_diffwave(monkeypatch):
    called = {}

    def fake_train_diffwave(checkpoint_dir=None):
        called["train"] = True
        called["checkpoint_dir"] = checkpoint_dir

    monkeypatch.setattr(diffwave_train, "train_diffwave", fake_train_diffwave)
    diffwave_train.main()
    assert called.get("train"), "train_diffwave was not called by main()"
    assert called.get("checkpoint_dir") == diffwave_train.DEFAULT_CHECKPOINT_DIR, (
        "Default checkpoint directory not used"
    )


def test_main_with_custom_out_dir(monkeypatch):
    called = {}
    custom_dir = "custom_checkpoints"

    def fake_train_diffwave(checkpoint_dir=None):
        called["train"] = True
        called["checkpoint_dir"] = checkpoint_dir

    # Save original argv
    original_argv = sys.argv.copy()
    try:
        # Set up command line arguments
        sys.argv = ["diffwave_train.py", "--out-dir", custom_dir]
        monkeypatch.setattr(diffwave_train, "train_diffwave", fake_train_diffwave)
        diffwave_train.main()
        assert called.get("train"), "train_diffwave was not called by main()"
        assert called.get("checkpoint_dir") == custom_dir, (
            "Custom checkpoint directory not used"
        )
    finally:
        # Restore original argv
        sys.argv = original_argv


def test_script_runs_without_error(monkeypatch):
    monkeypatch.setattr(
        diffwave_train, "train_diffwave", lambda checkpoint_dir=None: None
    )
    diffwave_train.main()  # Should not raise
