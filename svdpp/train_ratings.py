"""SVD++ sweep with 1024-rating batches and fixed FM hyperparameters.

python train_ratings.py --seed 11 --device cpu
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import ROOT, sha256, write_json
from src.experiment import context, digest, environment, git_info, validate_frame
from src.rating_training import HistoryOperator, RatingSVDPlusPlus, fit_ratings, tensors
from src.training import FixedOrderScorer, predict


def plots(output, records, grid):
    table=pd.DataFrame(records).sort_values("k")
    summary=table.rename(columns={"train_rmse":"train_rmse_mean", "validation_rmse":"validation_rmse_mean",
                                  "test_rmse":"test_rmse_mean"})
    summary["n_seeds"]=1
    summary.to_csv(output/"metrics_summary.csv",index=False,lineterminator="\n")
    figure,ax=plt.subplots(figsize=(10,5))
    ax.plot(summary.k,summary.validation_rmse_mean,"o-",label="Validation")
    if "test_rmse_mean" in summary:
        ax.plot(summary.k,summary.test_rmse_mean,"s-",label="Test (after freeze)")
    ax.set(xlabel="k (latent dimensions)",ylabel="RMSE",title=f"SVD++: fixed FM settings, seed {int(table.seed.iloc[0])}")
    ax.legend();ax.grid(alpha=.25);figure.tight_layout()
    for ext in ("png","pdf"):figure.savefig(output/f"rmse_vs_k.{ext}",dpi=180)
    ax.set_xscale("log");ax.set_xticks(grid,labels=[str(k) for k in grid],rotation=45)
    figure.tight_layout();figure.savefig(output/"rmse_vs_k_log.png",dpi=180);plt.close(figure)
    figure,axes=plt.subplots(math.ceil(len(grid)/3),3,figsize=(14,3.1*math.ceil(len(grid)/3)),squeeze=False)
    by_k={r["k"]:r for r in records}
    for ax,k in zip(axes.flat,grid):
        path=output/"runs"/f"k{k}"/"history.csv"
        if path.exists():
            history=pd.read_csv(path)
            ax.plot(history.epoch,history.train_objective,label="Train objective")
            ax.plot(history.epoch,history.train_mse,"--",label="Train MSE")
            if k in by_k:ax.axvline(by_k[k]["best_epoch"],color="gray",linestyle=":")
        ax.set(title=f"k={k}",xlabel="Epoch",ylabel="Train loss");ax.grid(alpha=.25)
    for ax in list(axes.flat)[len(grid):]:ax.set_visible(False)
    axes.flat[0].legend(fontsize=8);figure.tight_layout()
    figure.savefig(output/"final_loss_curves.png",dpi=150);plt.close(figure)
    for row in records:
        run=output/"runs"/f"k{row['k']}";h=pd.read_csv(run/"history.csv")
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(h.epoch,h.train_rmse,label="Train RMSE")
        axes[0].plot(h.epoch,h.validation_rmse,label="Validation RMSE")
        axes[1].plot(h.epoch,h.train_objective,label="MSE + bias L2 + factor L2")
        axes[1].plot(h.epoch,h.train_mse,"--",label="Train MSE")
        for ax in axes:
            ax.axvline(row["best_epoch"],color="gray",linestyle=":");ax.legend();ax.grid(alpha=.25);ax.set_xlabel("Epoch")
        fig.suptitle(f"SVD++ k={row['k']}, seed={row['seed']}; {row['updates_per_epoch']} Adam updates/epoch")
        fig.tight_layout();fig.savefig(run/"learning_and_loss.png",dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/svdpp-ml1m-ratings.json")
    parser.add_argument("--seed",type=int,default=11)
    parser.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    parser.add_argument("--k",type=int,nargs="+",help="Optional subset; test waits for the entire grid")
    parser.add_argument("--output",help="Separate output directory for this seed/protocol")
    args=parser.parse_args()
    if args.seed<0:parser.error("seed must be nonnegative")
    cfg=json.loads((ROOT/args.config).read_text(encoding="utf-8"));cfg["seed"]=args.seed
    output=ROOT/(args.output or f"results/svdpp-ml1m-ratings/seed{args.seed}")
    ks=args.k or cfg["k_grid"]
    if len(set(ks))!=len(ks) or not set(ks)<=set(cfg["k_grid"]):parser.error("k must be unique members of the configured grid")
    ctx=context("configs/svdpp-ml1m.json",args.device)
    torch.set_num_threads(cfg["threads"])
    device=ctx["device"]
    source_names=["train_ratings.py","src/rating_training.py","src/model.py","src/training.py",
                  "src/data.py","src/metrics.py","src/experiment.py"]
    protocol=dict(settings=cfg,split_id=ctx["meta"]["split"]["id"],data_sha256=ctx["meta"]["artifact_sha256"],
                  source_sha256={p:sha256(ROOT/p) for p in source_names},histories_checksum=ctx["histories"].checksum,
                  objective="MSE + reg_bias/N * (||bu||^2+||bi||^2) + reg_factors/N * (||P||^2+||Q||^2+||Y||^2)",
                  model="Canonical SVD++, fixed train mean, train-only histories, unclipped predictions",
                  selection="Fixed hyperparameters for every k; best epoch and best k by validation only",
                  batching="Fresh shuffle of all train ratings per epoch; one Adam step per rating batch",
                  history_sums="Recomputed at every Adam update; only immutable membership matrices are cached")
    fingerprint=digest(protocol)
    output.mkdir(parents=True,exist_ok=True)
    protocol_path=output/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8"))["fingerprint"]!=fingerprint:
            raise ValueError("Different code/settings/data in output: choose --output for a new experiment")
    else:
        write_json(protocol_path,dict(fingerprint=fingerprint,protocol=protocol,environment=environment(device),git=git_info()))
        with zipfile.ZipFile(output/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as archive:
            for p in source_names:archive.write(ROOT/p,p)
            archive.writestr("experiment_config.json",json.dumps(cfg,indent=2))
    operator=HistoryOperator(ctx["histories"],ctx["meta"]["n_items"])
    scorers=FixedOrderScorer(ctx["train"]),FixedOrderScorer(ctx["validation"])
    updates=math.ceil(len(ctx["train"])/cfg["batch_size"])
    print(f"START seed={args.seed} device={device} train={len(ctx['train'])} batch={cfg['batch_size']} updates/epoch={updates}",flush=True)
    records=[]
    def completed_records():
        result=[]
        for k in cfg["k_grid"]:
            path=output/"runs"/f"k{k}"/"metrics.json"
            if path.exists():result.append(json.loads(path.read_text(encoding="utf-8")))
        return result
    for k in ks:
        run=output/"runs"/f"k{k}"
        if (run/"metrics.json").exists():
            print(f"REUSE completed k={k}",flush=True)
        else:
            signature=digest([fingerprint,k,args.seed])
            _,metrics,_=fit_ratings(ctx["train"],ctx["validation"],ctx["histories"],ctx["meta"]["n_users"],
                                   ctx["meta"]["n_items"],cfg,device,k,args.seed,run,signature,operator,scorers)
        records=completed_records()
        plots(output,records,cfg["k_grid"])
        write_json(output/"status.json",dict(completed_k=[r["k"] for r in records],
                   pending_k=[k for k in cfg["k_grid"] if k not in {r["k"] for r in records}],
                   completed=False,seed=args.seed,updates_per_epoch=updates,test_evaluated=False))
    if {r["k"] for r in records}!=set(cfg["k_grid"]):
        print("Subset completed. Test remains closed until every k has completed.",flush=True);return
    frozen=output/"frozen.json"
    if not frozen.exists():
        selected=min(records,key=lambda r:(r["validation_rmse"],r["k"]))["k"]
        write_json(frozen,dict(fingerprint=fingerprint,selected_k_by_validation=selected,
                  models_sha256={str(k):sha256(output/"runs"/f"k{k}"/"model.pt") for k in cfg["k_grid"]},
                  epoch_cap_k=[r["k"] for r in records if r["hit_epoch_cap"]],test_used_for_selection=False))
    freeze=json.loads(frozen.read_text(encoding="utf-8"))
    if freeze["fingerprint"]!=fingerprint:raise ValueError("Frozen protocol mismatch")
    test=pd.read_csv(ctx["directory"]/"test.csv")
    validate_frame(test,ctx["meta"]["n_users"],ctx["meta"]["n_items"],"test")
    test_score=FixedOrderScorer(test);test_t=tensors(test,device)
    for record in records:
        k=record["k"];path=output/"runs"/f"k{k}"/"model.pt"
        if sha256(path)!=freeze["models_sha256"][str(k)]:raise ValueError("Frozen model changed")
        model=RatingSVDPlusPlus(ctx["meta"]["n_users"],ctx["meta"]["n_items"],k,float(ctx["train"].rating.mean()),
                               ctx["histories"],cfg["init_std"],operator).to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=False)["state"])
        record["test_rmse"]=test_score(predict(model,test_t,cfg["eval_batch_size"]))
    plots(output,records,cfg["k_grid"])
    pd.DataFrame(records).to_csv(output/"metrics_by_run.csv",index=False,lineterminator="\n")
    write_json(output/"status.json",dict(completed=True,completed_k=cfg["k_grid"],pending_k=[],seed=args.seed,
               selected_k_by_validation=freeze["selected_k_by_validation"],updates_per_epoch=updates,
               total_fit_seconds=sum(r["fit_seconds"] for r in records),test_evaluated=True))
    print(json.dumps(json.loads((output/"status.json").read_text()),indent=2),flush=True)

if __name__=="__main__":main()
