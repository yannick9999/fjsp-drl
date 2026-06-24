"""Run test.py repeatedly for multiple test sets and sampling modes.

For each data_path in DATA_PATHS, runs test.py once with sample=False (greedy, "_G")
and once with sample=True (sampling, "_S"), each time patching config.json beforehand
and renaming the resulting ./save/test_<timestamp> folder to ./save/<data_path>_<G|S>.

Already-finished combinations (target folder already exists) are skipped, so the
script can be safely re-run/resumed (e.g. after a crash on a cluster job).
"""
import json
import os
import subprocess
import sys

CONFIG_PATH = "./config.json"
SAVE_DIR = "./save"
DATA_PATHS = ["10010", "20010"]
# (sample, suffix)
# MODES = [(False, "G"), (True, "S")]
MODES = [(False, "G")]


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    original_config = load_config()
    try:
        for data_path in DATA_PATHS:
            for sample, suffix in MODES:
                target_name = "{0}_{1}".format(data_path, suffix)
                target_path = os.path.join(SAVE_DIR, target_name)
                if os.path.exists(target_path):
                    print("Skipping {0} (already exists)".format(target_name))
                    continue

                config = load_config()
                config["test_paras"]["data_path"] = data_path
                config["test_paras"]["sample"] = sample
                save_config(config)

                print("\n=== Running test: data_path={0}, sample={1} -> {2} ===".format(
                    data_path, sample, target_name))
                before = set(os.listdir(SAVE_DIR))
                result = subprocess.run([sys.executable, "test.py"])
                if result.returncode != 0:
                    print("test.py failed for {0} (exit code {1}), aborting.".format(
                        target_name, result.returncode))
                    return
                after = set(os.listdir(SAVE_DIR))
                new_dirs = [d for d in (after - before) if d.startswith("test_")]
                if len(new_dirs) != 1:
                    print("Warning: expected exactly one new save folder, found: {0}".format(new_dirs))
                    continue
                os.rename(os.path.join(SAVE_DIR, new_dirs[0]), target_path)
                print("Renamed {0} -> {1}".format(new_dirs[0], target_name))
    finally:
        save_config(original_config)
        print("\nRestored original config.json")


if __name__ == '__main__':
    main()
