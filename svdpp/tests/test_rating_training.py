import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.model import Histories,SVDPlusPlus
from src.rating_training import RatingSVDPlusPlus,fit_ratings,rating_batches,regularization,tensors

class RatingTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.frame=pd.DataFrame(dict(row_id=range(7),user_idx=[0,0,0,1,1,2,2],
              item_idx=[0,1,2,0,2,1,3],rating=[4.,2.,3.,5.,1.,2.,4.]))
        self.histories=Histories(self.frame,4)
    def test_predictions_gradients_and_adam_match_direct_svdpp(self):
        torch.manual_seed(11)
        sparse=RatingSVDPlusPlus(4,4,5,3.,self.histories)
        direct=SVDPlusPlus(4,4,5,3.,self.histories,.01*5**.25)
        direct.load_state_dict(sparse.state_dict())
        optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in (sparse,direct)]
        users,items,ratings=tensors(self.frame,torch.device('cpu'))
        for rows in (torch.tensor([0,3,6]),torch.tensor([1,2,4,5])):
            predictions=[]
            for model,opt in zip((sparse,direct),optimizers):
                opt.zero_grad();p=model(users[rows],items[rows]);predictions.append(p.detach())
                loss=(p-ratings[rows]).square().mean()+regularization(model,7,.01,10.)
                loss.backward()
            torch.testing.assert_close(*predictions,rtol=1e-6,atol=1e-6)
            for a,b in zip(sparse.parameters(),direct.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-7)
            for opt in optimizers:opt.step()
            for a,b in zip(sparse.parameters(),direct.parameters()):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)
        self.assertGreater(sparse.y.weight.grad.abs().sum().item(),0)
    def test_batches_cover_all_ratings_and_have_684_updates(self):
        batches=rating_batches(700146,1024,torch.Generator().manual_seed(11))
        self.assertEqual(len(batches),684)
        self.assertEqual(len(batches[-1]),754)
        torch.testing.assert_close(torch.cat(batches).sort().values,torch.arange(700146))
    def test_history_sum_uses_current_y_and_handles_empty_history(self):
        m=RatingSVDPlusPlus(4,4,2,3.,self.histories)
        users=torch.tensor([0,3]);before=m.profiles(users).detach().clone()
        with torch.no_grad():m.y.weight[0]+=1
        after=m.profiles(users).detach()
        torch.testing.assert_close(after[0]-before[0],torch.full((2,),1/np.sqrt(3)))
        torch.testing.assert_close(after[1],m.p.weight[3])
    def test_resume_matches_uninterrupted_updates_and_best_state(self):
        cfg=dict(max_epochs=3,patience=10,batch_size=3,learning_rate=.001,reg_bias=.01,
                 reg_factors=10.,init_std=.01,eval_batch_size=4,progress_every=1000)
        with tempfile.TemporaryDirectory() as tmp:
            a,am,ah=fit_ratings(self.frame,self.frame,self.histories,4,4,cfg,torch.device('cpu'),
                               2,11,Path(tmp)/'full','same')
            fit_ratings(self.frame,self.frame,self.histories,4,4,dict(cfg,max_epochs=1),torch.device('cpu'),
                        2,11,Path(tmp)/'resume','same')
            b,bm,bh=fit_ratings(self.frame,self.frame,self.histories,4,4,cfg,torch.device('cpu'),
                               2,11,Path(tmp)/'resume','same')
            self.assertEqual(am['optimizer_updates'],9)
            self.assertEqual(bm['optimizer_updates'],9)
            for name in a.state_dict():torch.testing.assert_close(a.state_dict()[name],b.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(am['validation_rmse'],bm['validation_rmse'])
            self.assertEqual([r['train_objective'] for r in ah],[r['train_objective'] for r in bh])


class RatingSweepTests(unittest.TestCase):
    def test_test_stays_closed_until_entire_grid_is_frozen(self):
        import json
        import shutil
        import train_ratings
        from unittest.mock import patch
        from src.data import ROOT as source_root
        torch.set_num_threads(1)
        frame=pd.DataFrame(dict(row_id=range(7),user_idx=[0,0,0,1,1,2,2],
                  item_idx=[0,1,2,0,2,1,3],rating=[4.,2.,3.,5.,1.,2.,4.]))
        cfg=dict(k_grid=[1,2],seed=11,max_epochs=1,patience=10,batch_size=3,
                 learning_rate=.001,reg_bias=.01,reg_factors=10.,init_std=.01,
                 eval_batch_size=4,threads=1,device='cpu',progress_every=1000)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'configs').mkdir();data=root/'data';data.mkdir()
            (root/'configs/svdpp-ml1m-ratings.json').write_text(json.dumps(cfg))
            for name in ['train_ratings.py','src/rating_training.py','src/model.py','src/training.py',
                         'src/data.py','src/metrics.py','src/experiment.py']:
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source_root/name,dest)
            ctx=dict(train=frame,validation=frame.assign(row_id=frame.row_id+10),
                     histories=Histories(frame,4),meta={'n_users':4,'n_items':4,'split':{'id':'tiny'},'artifact_sha256':{}},
                     device=torch.device('cpu'),directory=data)
            with patch.object(train_ratings,'ROOT',root),patch.object(train_ratings,'context',return_value=ctx), \
                 patch.object(train_ratings,'git_info',return_value={}), \
                 patch('sys.argv',['train_ratings.py','--k','1']):
                train_ratings.main()
            output=root/'results/svdpp-ml1m-ratings/seed11'
            self.assertFalse((output/'frozen.json').exists())
            self.assertFalse(json.loads((output/'status.json').read_text())['test_evaluated'])
            # No test file existed during the first invocation, so reading test early would fail.
            frame.assign(row_id=frame.row_id+20).to_csv(data/'test.csv',index=False)
            read_csv=pd.read_csv
            def guarded_read(path,*args,**kwargs):
                if Path(path)==data/'test.csv':
                    self.assertTrue((output/'frozen.json').exists())
                return read_csv(path,*args,**kwargs)
            with patch.object(train_ratings,'ROOT',root),patch.object(train_ratings,'context',return_value=ctx), \
                 patch.object(train_ratings,'git_info',return_value={}), \
                 patch('sys.argv',['train_ratings.py','--k','2']),patch.object(pd,'read_csv',side_effect=guarded_read):
                train_ratings.main()
            status=json.loads((output/'status.json').read_text())
            self.assertTrue(status['completed']);self.assertTrue(status['test_evaluated'])
            self.assertEqual(status['completed_k'],[1,2])
            self.assertEqual(status['updates_per_epoch'],3)
            self.assertEqual(len(pd.read_csv(output/'metrics_by_run.csv')),2)
