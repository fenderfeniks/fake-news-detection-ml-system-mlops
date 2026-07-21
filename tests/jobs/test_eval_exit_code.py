from unittest.mock import MagicMock, patch

import pytest


def test_eval_exits_1_when_metric_below_threshold():
    """eval.py должен завершиться с кодом 1 если test_f1 < drift_threshold."""

    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, *args: {
        "ckpt_path": None,
        "drift_threshold": 0.9,
    }.get(key)

    mock_trainer = MagicMock()
    mock_trainer.test.return_value = [{"test_f1": 0.75, "test_acc": 0.80}]

    with pytest.raises(SystemExit) as exc_info:
        from src.eval import _check_drift  # вынести логику в отдельную функцию

        _check_drift(
            results=[{"test_f1": 0.75, "test_acc": 0.80}],
            drift_threshold=0.9,
            metric_key="test_f1",
        )

    assert exc_info.value.code == 1


def test_eval_does_not_exit_when_metric_above_threshold():
    """eval.py не должен падать если метрика выше порога."""
    from src.eval import _check_drift

    # Не должно бросить SystemExit
    _check_drift(
        results=[{"test_f1": 0.95, "test_acc": 0.92}],
        drift_threshold=0.9,
        metric_key="test_f1",
    )


def test_eval_warns_when_metric_key_missing():
    """Если ключ метрики не найден в результатах — warning, не падение."""

    from src.eval import _check_drift

    with patch("logging.Logger.warning") as mock_warn:
        _check_drift(
            results=[{"test_acc": 0.80}],  # нет test_f1
            drift_threshold=0.9,
            metric_key="test_f1",
        )
        mock_warn.assert_called()
