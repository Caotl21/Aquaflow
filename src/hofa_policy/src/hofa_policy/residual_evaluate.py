"""Deterministic evaluation of baseline, TD3 residual and SAC residual."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import TD3, SAC
from hofa_mpc_ros.tracking_plots import generate_run_plots, generate_comparison_plots
from .residual_env import ResidualTrackingEnv, read_config, write_csv
from .rl_backend import sha256, workspace_root
from .residual_train import tensor_hash


def validate_model(model_path, config, algo):
    model_path=Path(model_path).resolve()
    manifest_path=model_path.parent/"training.json"
    if not manifest_path.exists():
        # Periodic checkpoints live one directory below the training manifest.
        manifest_path=model_path.parent.parent/"training.json"
    manifest=json.loads(manifest_path.read_text())
    # Reward-only experiments must be evaluated against a shared objective.
    # Observation/action/physics settings still have to match exactly.
    objective_keys = {"training", "reward", "tracking_reward", "potential_reward",
                      "regularization", "comparison_objective", "validation"}
    env_config={k:v for k,v in config.items() if k not in objective_keys}
    trained_env_config={k:v for k,v in manifest["config"].items() if k not in objective_keys}
    if manifest["algo"]!=algo or trained_env_config!=env_config:
        raise ValueError("model algorithm/config does not match evaluation contract")
    provenance_path=manifest_path.parent/"train_env/provenance.json"
    provenance=json.loads(provenance_path.read_text())
    checkpoint=Path(config["baseline_checkpoint"])
    if not checkpoint.is_absolute(): checkpoint=workspace_root()/checkpoint
    if sha256(checkpoint)!=provenance["baseline_sha256"]:
        raise ValueError("frozen baseline checkpoint has changed since training")


def evaluate(algo, model_path, config, output, random_cases=0, cases=None, make_plots=True):
    output=Path(output)
    if model_path:
        validate_model(model_path,config,algo)
        model=(TD3 if algo=="td3" else SAC).load(model_path,device="cpu")
    else:
        model=None
    env=ResidualTrackingEnv(config,output,training=False,record_steps=True)
    results=[]
    try:
        frozen=tensor_hash(env.policy.model)
        if cases is None:
            cases=[]
            for i,route in enumerate(config["routes"]):
                cases.append((route,False,20000+i,"fixed_"+route))
                for j in range(random_cases):
                    cases.append((route,True,21000+100*i+j,"random_%s_%02d" % (route,j)))
        for route,randomize,seed,label in cases:
            obs,_=env.reset(seed=seed,options=dict(route=route,randomize=randomize))
            done=False
            while not done:
                action=np.zeros(3,dtype=np.float32) if model is None else model.predict(obs,deterministic=True)[0]
                obs,_,terminated,truncated,info=env.step(action)
                done=terminated or truncated
            metrics=info["episode_metrics"]
            run_dir=output/("episode_%06d" % env.episode)
            if make_plots:
                generate_run_plots(run_dir,label=algo+" / "+label)
            metrics.update(label=label,algo=algo,seed=seed,randomized=randomize,run_dir=str(run_dir.resolve()))
            results.append(metrics)
            print(json.dumps({key:metrics[key] for key in ["algo","label","finish_reason","duration_s","reward"]}),flush=True)
        if frozen!=tensor_hash(env.policy.model):
            raise AssertionError("frozen baseline mutated during evaluation")
    finally:
        env.close()
    (output/"evaluation.json").write_text(json.dumps(results,indent=2)+"\n")
    return results


def compare(args):
    config=read_config(args.config)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    methods=[("dagger",None)]
    if args.td3_model: methods.append(("td3",args.td3_model))
    if args.sac_model: methods.append(("sac",args.sac_model))
    results=[]
    for algo,path in methods:
        results.extend(evaluate(algo,path,config,output/algo,args.random_cases))
    rows=[]
    for r in results:
        row={key:r[key] for key in ("algo","label","seed","finish_reason","duration_s","completion_ratio","reward","run_dir")}
        for name,key in [("cross_track_rmse_m","cross_track_error_m"),
                         ("yaw_rmse_deg","yaw_error_deg"),
                         ("turn_yaw_rmse_deg","turn_yaw_error_deg"),
                         ("turn_yaw_rate_rmse_radps","turn_yaw_rate_error_radps")]:
            row[name]=(r.get(key) or {}).get("rmse")
        rows.append(row)
    (output/"comparison.json").write_text(json.dumps(dict(config=config,runs=rows),indent=2)+"\n")
    write_csv(output/"comparison.csv",rows)
    for label in sorted(set(r["label"] for r in results)):
        generate_comparison_plots([(r["algo"],r["run_dir"]) for r in results if r["label"]==label],
                                  output/"plots"/label)
    print("Comparison artifacts: %s" % output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output",required=True)
    parser.add_argument("--td3-model")
    parser.add_argument("--sac-model")
    parser.add_argument("--random-cases",type=int,default=0,
                        help="additional seeded held-out variations per route")
    args=parser.parse_args()
    if args.random_cases<0: parser.error("random cases must be nonnegative")
    compare(args)


if __name__=="__main__":
    main()
