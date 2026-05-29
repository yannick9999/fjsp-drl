"""
Compare Song's pretrained model vs. your trained model on the same 100 dev instances.
Run: python compare_models.py
"""
import copy
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import PPO_model
from env.fjsp_env import FJSPEnv

# ── Config ────────────────────────────────────────────────────────────────────
MY_MODEL_PATH   = "./save/train_20260529_103832/save_best_10_5_810.pt"
SONG_MODEL_PATH = "./results/save_10_5.pt"
DATA_DIR        = "./data_test/1005/"
MY_TRAINING_AVE = "./save/train_20260529_103832/training_ave_20260529_103832.xlsx"
MY_TRAINING_100 = "./save/train_20260529_103832/training_100_20260529_103832.xlsx"
# ─────────────────────────────────────────────────────────────────────────────

def load_config():
    with open("./config.json") as f:
        cfg = json.load(f)
    return cfg

def build_env(env_paras, data_dir):
    files = sorted(os.listdir(data_dir))
    paths = [data_dir + f for f in files if f.endswith(".fjs")][:100]
    env_paras_copy = copy.deepcopy(env_paras)
    env_paras_copy["batch_size"] = len(paths)
    env = FJSPEnv(case=paths, env_paras=env_paras_copy, data_source="file")
    return env, len(paths)

def run_model(env, model):
    memory = PPO_model.Memory()
    state  = env.state
    done   = False
    dones  = env.done_batch
    while not done:
        with torch.no_grad():
            actions = model.policy_old.act(state, memory, dones,
                                           flag_sample=False, flag_train=False)
        state, _, dones = env.step(actions)
        done = dones.all()
    ok = env.validate_gantt()[0]
    if not ok:
        print("  WARNING: Scheduling Error detected!")
    makespans = copy.deepcopy(env.makespan_batch)
    env.reset()
    return makespans.cpu().numpy()

def load_model(path, model_paras, train_paras):
    model = PPO_model.PPO(model_paras, train_paras, num_envs=100)
    ckpt  = torch.load(path, map_location="cpu")
    model.policy.load_state_dict(ckpt)
    model.policy_old.load_state_dict(ckpt)
    model.policy.eval()
    model.policy_old.eval()
    return model

def main():
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    cfg = load_config()
    env_paras   = cfg["env_paras"]
    model_paras = cfg["model_paras"]
    train_paras = cfg["train_paras"]
    env_paras["device"]   = device
    model_paras["device"] = device
    model_paras["actor_in_dim"]  = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    # ── Run Song's model ──────────────────────────────────────────────────────
    print("Loading Song's model ...")
    song_model = load_model(SONG_MODEL_PATH, model_paras, train_paras)
    env, n = build_env(env_paras, DATA_DIR)
    print(f"  Running on {n} instances ...")
    song_makespans = run_model(env, song_model)
    print(f"  Song  avg makespan: {song_makespans.mean():.2f}  (std {song_makespans.std():.2f})")

    # ── Run your model ────────────────────────────────────────────────────────
    print("Loading your model (best checkpoint iter 810) ...")
    my_model = load_model(MY_MODEL_PATH, model_paras, train_paras)
    env, _ = build_env(env_paras, DATA_DIR)
    print(f"  Running on {n} instances ...")
    my_makespans = run_model(env, my_model)
    print(f"  Yours avg makespan: {my_makespans.mean():.2f}  (std {my_makespans.std():.2f})")

    # ── Print summary ─────────────────────────────────────────────────────────
    diff_pct = (my_makespans.mean() - song_makespans.mean()) / song_makespans.mean() * 100
    print(f"\n  Gap (yours vs Song): {diff_pct:+.2f}%")
    if abs(diff_pct) <= 5:
        print("  -> Results are within 5% — implementation looks CORRECT.")
    else:
        print("  -> Gap > 5%. Could be due to fewer training iterations or different random seed.")

    # ── Plot 1: Learning curve + Song baseline ────────────────────────────────
    import pandas as pd
    df_ave = pd.read_excel(MY_TRAINING_AVE)
    df_100 = pd.read_excel(MY_TRAINING_100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FJSP 10×5 — Your Training vs. Song's Model", fontsize=13, fontweight="bold")

    # Left: learning curve
    ax = axes[0]
    ax.plot(df_ave["iterations"], df_ave["res"], color="steelblue", linewidth=1.5, label="Your model (val avg)")

    # Rolling average for clarity
    roll = df_ave["res"].rolling(window=5, center=True).mean()
    ax.plot(df_ave["iterations"], roll, color="navy", linewidth=2.5, linestyle="--", label="Rolling avg (w=5)")

    song_line = song_makespans.mean()
    ax.axhline(song_line, color="tomato", linewidth=2, linestyle="-", label=f"Song baseline ({song_line:.1f})")
    ax.axvline(810, color="green", linewidth=1.5, linestyle=":", label="Best checkpoint (iter 810)")

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Avg Makespan (100 dev instances)")
    ax.set_title("Learning Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: per-instance comparison (sorted)
    ax2 = axes[1]
    idx = np.argsort(song_makespans)
    x   = np.arange(n)
    ax2.plot(x, song_makespans[idx], color="tomato",    linewidth=1.2, label=f"Song  (avg {song_makespans.mean():.1f})")
    ax2.plot(x, my_makespans[idx],   color="steelblue", linewidth=1.2, label=f"Yours (avg {my_makespans.mean():.1f})")
    ax2.set_xlabel("Instance (sorted by Song makespan)")
    ax2.set_ylabel("Makespan")
    ax2.set_title("Per-Instance Comparison (100 dev instances)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "comparison_10x5.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()

if __name__ == "__main__":
    main()
