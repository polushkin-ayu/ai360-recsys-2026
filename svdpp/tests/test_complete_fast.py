"""Continuation checks: preserve last optimizer/RNG state and immutable source."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

import complete_fast as finish
from src.data import sha256
from src.experiment import digest, read_json, trial_id, write_json
from src.model import Histories
from src.training import save_torch


class CompletionTests(unittest.TestCase):
    def test_completion_resumes_last_state_reuses_finished_runs_and_freezes_before_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            cfg = dict(k_grid=[1], learning_rates=[.003, .03], regularizations=[0., 1.],
                       tuning_strategy=dict(screening_epochs=2, candidates_per_learning_rate=1),
                       max_epochs=6, patience=2, batch_users=2, eval_batch_size=10)
            frame = pd.DataFrame(dict(row_id=[0, 1], user_idx=[0, 1], item_idx=[0, 1], rating=[4., 2.]))
            ctx = dict(cfg=cfg, protocol={"source_sha256": {}}, output=root / "completed",
                       checkpoints=root / "weights", train=frame, validation=frame,
                       histories=Histories(frame, 2), device=torch.device("cpu"),
                       meta=dict(n_users=2, n_items=2, split={"id": "fixture"}))
            originals = {}
            for stage, pairs in (("screen", [(.003, 0.), (.003, 1.), (.03, 0.), (.03, 1.)]),
                                 ("tune", [(.003, 0.), (.03, 1.)])):
                for lr, reg in pairs:
                    limited = (stage, lr, reg) in (("screen", .003, 0.), ("tune", .03, 1.))
                    epoch = (1 if limited else 2) if stage == "screen" else (3 if limited else 6)
                    identifier = trial_id(stage, 1, lr, reg, 11)
                    sig = digest(["source-fingerprint", stage, 1, lr, reg, 11])
                    metrics = dict(k=1, learning_rate=lr, regularization=reg, seed=11,
                                   epochs_run=epoch, best_epoch=epoch, time_limited=limited,
                                   train_rmse=.8, validation_rmse=.9+lr, fit_seconds=.1,
                                   hit_epoch_cap=not limited)
                    history = [dict(epoch=epoch, train_rmse=.8, validation_rmse=.9+lr,
                                    train_mse=.64, train_objective=.65)]
                    last, best, hist = (source/(identifier+suffix) for suffix in ("_last.pt", "_best.pt", ".csv"))
                    save_torch(last, dict(signature=sig, epoch=epoch, model={"weight":torch.tensor([100.])},
                                         optimizer={"step":77}, rng={"marker":123}, history=history))
                    save_torch(best, dict(signature=sig, fingerprint="source-fingerprint",
                                         state={"weight":torch.tensor([0.])}, metrics=metrics))
                    pd.DataFrame(history).to_csv(hist,index=False)
                    record = dict(id=identifier, stage=stage, signature=sig, metrics=metrics,
                                  last_checkpoint=str(last.relative_to(root)), checkpoint=str(best.relative_to(root)),
                                  files={str(p.relative_to(root)):sha256(p) for p in (last,best,hist)})
                    originals[identifier] = record
            write_json(source / "frozen.json", dict(test_complete=True))
            source_hashes = {p:sha256(p) for p in source.iterdir() if p.is_file()}
            calls = []
            class Model:
                def state_dict(self):
                    return {"weight":torch.tensor([101.])}
            def fake_fit(*args, **kwargs):
                self.assertIsNone(kwargs["deadline"])
                state = torch.load(kwargs["resume_path"], weights_only=False)
                self.assertEqual(state["model"]["weight"].item(),100.)  # Never restart from best=0.
                self.assertEqual(state["optimizer"]["step"],77)
                self.assertEqual(state["rng"]["marker"],123)
                self.assertEqual(state["signature"],kwargs["signature"])
                calls.append(state["epoch"])
                cap = kwargs["max_epochs"]
                m = dict(k=1, learning_rate=args[8], regularization=args[9], seed=11,
                         epochs_run=cap, best_epoch=cap, time_limited=False, hit_epoch_cap=True,
                         train_rmse=.8, validation_rmse=.85, fit_seconds=.2)
                hist = state["history"] + [dict(epoch=cap, train_rmse=.8, validation_rmse=.85,
                                               train_mse=.64, train_objective=.65)]
                save_torch(kwargs["resume_path"],dict(signature=kwargs["signature"],epoch=cap))
                return Model(), m, hist
            def test_after_freeze(ctx):
                frozen = read_json(ctx["output"] / "frozen.json")
                self.assertFalse(frozen["test_started"])
                self.assertTrue(frozen["source_test_already_evaluated"])
                self.assertEqual(len(frozen["runs"]),1)
                self.assertFalse(frozen["runs"][0]["metrics"]["time_limited"])
                self.assertEqual(frozen["runs"][0]["metrics"]["learning_rate"],.03)
            saved_source = dict(fingerprint="source-fingerprint",protocol=dict(fast_seed=11))
            with patch("complete_fast.ROOT",root), patch("src.experiment.ROOT",root), \
                    patch("complete_fast.fit_one",side_effect=fake_fit), \
                    patch("complete_fast.evaluate_test",side_effect=test_after_freeze), \
                    patch("complete_fast.plots"):
                finish.run(ctx,source,saved_source,originals)
                self.assertEqual(calls,[1,3])  # Only the two time-limited trials train.
                status=read_json(ctx["output"] / "completion_status.json")
                self.assertEqual(status["additional_epochs"],4)
                self.assertEqual(status["remaining_time_limited_runs"],0)
                self.assertTrue(status["complete"])
                self.assertIsNone(status["wall_time_limit"])
                for p, expected in source_hashes.items():
                    self.assertEqual(sha256(p),expected)
                original=next(r for r in originals.values() if r["metrics"]["time_limited"])
                fp=digest(ctx["protocol"])
                with patch("complete_fast.fit_one",side_effect=AssertionError("Must reuse completed continuation")):
                    finish.complete_trial(ctx,original,fp,"source-fingerprint",None)
                hist=ctx["output"] / "histories" / (original["id"]+".csv")
                hist.write_text("tampered",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"Saved artifact changed"):
                    finish.complete_trial(ctx,original,fp,"source-fingerprint",None)


if __name__ == "__main__":
    unittest.main()
