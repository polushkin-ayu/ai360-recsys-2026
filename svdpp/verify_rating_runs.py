"""Audit a completed rating-batch sweep without any gradient updates."""
import json
import hashlib
import math
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import ROOT,sha256,write_json
from src.experiment import context,digest
from src.rating_training import HistoryOperator,RatingSVDPlusPlus,tensors
from src.training import FixedOrderScorer,predict

def require(value,message):
    if not value:raise RuntimeError(message)

def main():
    out=ROOT/'results/svdpp-ml1m-ratings-active/seed11'
    status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    require(status['completed'] and status['test_evaluated'],'Study is incomplete')
    stored=json.loads((out/'protocol.json').read_text(encoding='utf-8'));protocol=stored['protocol'];cfg=protocol['settings']
    require(digest(protocol)==stored['fingerprint'],'Protocol fingerprint mismatch')
    with zipfile.ZipFile(out/'source_snapshot.zip') as archive:
        require(json.loads(archive.read('experiment_config.json'))==cfg,'Snapshot settings mismatch')
        for name,expected in protocol['source_sha256'].items():
            require(sha256(ROOT/name)==expected,f'Current source differs: {name}')
            require(hashlib.sha256(archive.read(name)).hexdigest()==expected,f'Source snapshot differs: {name}')
    frozen=json.loads((out/'frozen.json').read_text(encoding='utf-8'))
    require(frozen['fingerprint']==stored['fingerprint'],'Freeze fingerprint mismatch')
    require(frozen['test_used_for_selection'] is False,'Test used for selection')
    ctx=context('configs/svdpp-ml1m.json','cpu');torch.set_num_threads(cfg['threads'])
    require(ctx['meta']['split']['id']==protocol['split_id'],'Split mismatch')
    require(ctx['meta']['artifact_sha256']==protocol['data_sha256'],'Data hash mismatch')
    require(not(set(ctx['train'].row_id)&set(ctx['validation'].row_id)),'Train/validation overlap')
    require(ctx['histories'].checksum==protocol['histories_checksum'],'Histories mismatch')
    test=pd.read_csv(ctx['directory']/'test.csv');test_t=tensors(test,torch.device('cpu'));val_t=tensors(ctx['validation'],torch.device('cpu'))
    test_score,val_score=FixedOrderScorer(test),FixedOrderScorer(ctx['validation'])
    require(not(set(test.row_id)&set(ctx['train'].row_id)),'Train/test overlap')
    require(not(set(test.row_id)&set(ctx['validation'].row_id)),'Validation/test overlap')
    table=pd.read_csv(out/'metrics_by_run.csv')
    require(set(table.k)==set(cfg['k_grid']) and len(table)==len(cfg['k_grid']),'Incomplete grid')
    operator=HistoryOperator(ctx['histories'],ctx['meta']['n_items']);maximum_error=0.;checked=0;parameter_counts={}
    for row in table.to_dict('records'):
        k=int(row['k']);run=out/'runs'/f'k{k}';history=pd.read_csv(run/'history.csv')
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        best=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
        require(saved['signature']==digest([stored['fingerprint'],k,int(row['seed'])]),f'Signature k={k}')
        require(1<=row['epochs_run']<=25,f'Epoch cap k={k}')
        require(list(history.epoch)==list(range(1,int(row['epochs_run'])+1)),f'Epoch coverage k={k}')
        require((history.updates_in_epoch==684).all(),f'Updates/epoch k={k}')
        require(np.array_equal(history.optimizer_updates,history.epoch*684),f'Cumulative updates k={k}')
        steps={int(s['step'].item()) for s in saved['optimizer']['state'].values()}
        require(steps=={int(row['epochs_run'])*684},f'Actual Adam steps k={k}')
        require(saved['updates']==int(row['epochs_run'])*684,f'Checkpoint updates k={k}')
        for field in ('learning_rate','reg_bias','reg_factors','init_std','batch_size'):
            require(row[field]==cfg[field],f'Fixed setting {field} k={k}')
        chosen=history.loc[history.validation_rmse.idxmin()]
        require(int(chosen.epoch)==int(row['best_epoch']),f'Best epoch k={k}')
        require(abs(float(chosen.validation_rmse)-float(row['validation_rmse']))<1e-7,f'Best validation metric k={k}')
        for name,weight in saved['best_state'].items():
            torch.testing.assert_close(weight,best['state'][name],rtol=0,atol=0)
        require(sha256(run/'model.pt')==frozen['models_sha256'][str(k)],f'Frozen weight hash k={k}')
        model=RatingSVDPlusPlus(ctx['meta']['n_users'],ctx['meta']['n_items'],k,float(ctx['train'].rating.mean()),ctx['histories'],cfg['init_std'],operator)
        require(len(saved['optimizer']['state'])==len(list(model.parameters())),f'Missing Adam states k={k}')
        parameter_counts[str(k)]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.load_state_dict(best['state'])
        val=val_score(predict(model,val_t,cfg['eval_batch_size']));score=test_score(predict(model,test_t,cfg['eval_batch_size']))
        error=max(abs(val-row['validation_rmse']),abs(score-row['test_rmse']));maximum_error=max(maximum_error,error)
        require(error<1e-7,f'Recomputed metric k={k}: {error}')
        require((run/'learning_and_loss.png').exists(),f'Missing plot k={k}')
        checked+=1
    best=table.sort_values(['validation_rmse','k']).iloc[0]
    require(int(best.k)==frozen['selected_k_by_validation']==status['selected_k_by_validation'],'Selected k mismatch')
    for name in ['rmse_vs_k.png','rmse_vs_k.pdf','rmse_vs_k_log.png','final_loss_curves.png']:
        require((out/name).exists(),f'Missing plot: {name}')
    hashes={str(p.relative_to(out)).replace('\\','/'):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='verification.json'}
    result=dict(passed=True,models_checked=checked,updates_per_epoch=684,max_epochs=25,
                total_optimizer_updates=int(table.optimizer_updates.sum()),maximum_recomputed_metric_difference=maximum_error,
                selected_k_by_validation=int(best.k),source_snapshot_checked=True,frozen_weights_checked=True,
                histories_and_adam_steps_checked=True,trainable_parameter_counts=parameter_counts,
                data_counts=dict(train=len(ctx['train']),validation=len(ctx['validation']),test=len(test),n_users=ctx['meta']['n_users'],n_items=ctx['meta']['n_items']),
                artifact_sha256=hashes)
    write_json(out/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))

if __name__=='__main__':main()
