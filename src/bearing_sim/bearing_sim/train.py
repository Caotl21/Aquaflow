"""Schema-checked behavior cloning with independent validation selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import BearingPolicy, policy_loss
from .records import FIELDS, HORIZON, SCHEMA, file_hash, source_hashes, write_json, scene_id

def load_dataset(path,split):
    path=Path(path); meta=json.loads(path.with_suffix('.json').read_text())
    if meta.get('schema')!=SCHEMA or meta.get('split')!=split or meta.get('data_sha256')!=file_hash(path): raise ValueError('dataset schema/hash mismatch')
    if any(s['scene_id']!=scene_id(s) for s in meta['scenes']): raise ValueError('scene ID mismatch')
    with np.load(path,allow_pickle=False) as a: data={k:a[k].copy() for k in (*FIELDS,'expert_action','expert_mode','scene_episode')}
    n=len(data['tokens']); expected={'track_mask':(n,data['tokens'].shape[1],data['tokens'].shape[2]),'frame_mask':(n,data['tokens'].shape[1]),'age':(n,data['tokens'].shape[1]),'ego_history':(n,data['tokens'].shape[1],7),'velocity':(n,3),'goal':(n,2),'expert_action':(n,HORIZON,3),'expert_mode':(n,),'scene_episode':(n,)}
    if data['tokens'].ndim!=4 or data['tokens'].shape[-1]!=7 or any(data[k].shape!=v for k,v in expected.items()): raise ValueError('dataset shape mismatch')
    if any(not np.isfinite(v).all() for v in data.values()): raise ValueError('nonfinite dataset')
    if data['track_mask'].dtype!=np.bool_ or data['frame_mask'].dtype!=np.bool_ or not np.isin(data['expert_mode'],[0,1,2]).all(): raise ValueError('invalid masks or modes')
    if np.any(data['age']<0) or np.linalg.norm(data['expert_action'][...,:2],axis=-1).max()>.20001: raise ValueError('invalid labels')
    return TensorDataset(*[torch.from_numpy(data[k]) for k in (*FIELDS,'expert_action','expert_mode')]),meta

def batch_loss(model,batch,device):
    inputs={k:batch[i].to(device) for i,k in enumerate(FIELDS)}
    return policy_loss(model(**inputs),batch[len(FIELDS)].to(device),batch[len(FIELDS)+1].to(device))

@torch.no_grad()
def val_loss(model,loader,device):
    model.eval(); total=0.
    for batch in loader: total+=float(batch_loss(model,batch,device))*len(batch[0])
    return total/len(loader.dataset)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--val',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--device',default='cpu'); p.add_argument('--seed',type=int,default=42); p.add_argument('--threads',type=int,default=2); a=p.parse_args()
    if min(a.epochs,a.batch_size,a.threads)<1 or a.lr<=0: p.error('invalid optimization settings')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.use_deterministic_algorithms(True)
    train,tm=load_dataset(a.train,'train'); val,vm=load_dataset(a.val,'val'); tids={s['scene_id'] for s in tm['scenes']}; vids={s['scene_id'] for s in vm['scenes']}
    if tids&vids: raise ValueError('train/validation scene overlap')
    c=tm['scenes'][0]['config']; limits={k:c[k] for k in ('dt','max_speed','max_yaw_rate','max_accel','max_yaw_accel')}; model=BearingPolicy(**limits).to(a.device)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True); vloader=DataLoader(val,batch_size=a.batch_size)
    batch_loss(model,next(iter(loader)),a.device); a.output.mkdir(parents=True,exist_ok=False)
    used_scene_ids=sorted(tids|vids)
    provenance=dict(schema=SCHEMA,seed=a.seed,train_sha256=file_hash(a.train),val_sha256=file_hash(a.val),
                    train_scene_ids=sorted(tids),val_scene_ids=sorted(vids),used_scene_ids=used_scene_ids,
                    source_sha256=source_hashes(),parameters=sum(x.numel() for x in model.parameters()))
    write_json(a.output/'run.json',provenance)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); best=float('inf'); logs=[]
    for epoch in range(a.epochs):
        model.train(); total=0.
        for batch in loader:
            opt.zero_grad(set_to_none=True); loss=batch_loss(model,batch,a.device); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step(); total+=float(loss.detach())*len(batch[0])
        vl=val_loss(model,vloader,a.device); row=dict(epoch=epoch+1,train_loss=total/len(train),val_loss=vl); logs.append(row); payload=dict(model=model.state_dict(),config=model.config,provenance=provenance,epoch=epoch+1,val_loss=vl)
        if vl<best: best=vl; torch.save(payload,a.output/'best.pt')
        torch.save(payload,a.output/'last.pt'); write_json(a.output/'learning_curve.json',logs); print(json.dumps(row),flush=True)
    selected=torch.load(a.output/'best.pt',map_location=a.device,weights_only=True); model.load_state_dict(selected['model']); reload=val_loss(model,vloader,a.device)
    if not np.isclose(reload,best,rtol=1e-6): raise RuntimeError('checkpoint reload mismatch')
    write_json(a.output/'summary.json',dict(best_val_loss=best,reload_val_loss=reload,best_epoch=selected['epoch'],train_samples=len(train),val_samples=len(val)))

if __name__=='__main__': main()
