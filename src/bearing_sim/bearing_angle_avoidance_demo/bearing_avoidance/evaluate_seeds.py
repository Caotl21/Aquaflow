#!/usr/bin/env python3
"""固定参数批量测试，不按结果挑选场景。"""
from pathlib import Path
import argparse,csv,json
from avoidance_demo import Config,run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--output',default='benchmark');p.add_argument('--particles',type=int,default=1000)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for seed in a.seeds:
        summaries.append(run(Config(seed=seed,particles=a.particles),out/f'seed_{seed}',plots=False))
    summary=dict(seeds=a.seeds,successes=sum(x['status']=='reached' for x in summaries),
                 collisions=sum(x['status']=='collision' for x in summaries),
                 stalls=sum(x['status']=='stalled' for x in summaries),
                 timeouts=sum(x['status']=='timeout' for x in summaries),runs=summaries)
    (out/'benchmark.json').write_text(json.dumps(summary,indent=2))
    fields=['seed','status','steps','path_length_m','minimum_true_surface_clearance_m','goal_distance_m','stops','discovered_count']
    with (out/'benchmark.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    print('BENCHMARK:',json.dumps({k:v for k,v in summary.items() if k!='runs'}))
