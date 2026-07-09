#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# Edit this dict to define the hyperparameters and values to sweep.
# Plain keys are applied to both model.<key> and model.<dataset>.<key>
# when the dataset-specific section exists.
SEARCH_SPACE = {
    "hyper_weight": [0.5, 1.0],
    "hyper_cl_weight": [1.0e-4, 1.0e-3],
    "hyper_cl_temperature": [0.1],
}

# Optional fixed overrides. Use dotted keys for exact config paths.
# Example: {"train.save_model": False, "train.patience": 5}
FIXED_OVERRIDES = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a grid search and record final test results.")
    parser.add_argument("--model", default="dccf_int", help="Base model config name under encoder/config/modelconf.")
    parser.add_argument("--dataset", default="amazon", help="Dataset name passed to train_encoder.py.")
    parser.add_argument("--device", default="cuda", help="Device passed to train_encoder.py.")
    parser.add_argument("--cuda", default="0", help="CUDA id passed to train_encoder.py.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed passed to train_encoder.py.")
    parser.add_argument("--space", default=None, help="JSON search space, e.g. '{\"hyper_weight\":[0,1]}'.")
    parser.add_argument("--space-file", default=None, help="JSON/YAML file containing the search space.")
    parser.add_argument("--output", default=None, help="CSV path. Default: encoder/sweep_results/<model>_<dataset>_<time>.csv")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run training.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip combinations before this index.")
    parser.add_argument("--max-runs", type=int, default=None, help="Run at most this many combinations.")
    parser.add_argument("--dry-run", action="store_true", help="Print combinations without training.")
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_search_space(args):
    if args.space:
        space = json.loads(args.space)
    elif args.space_file:
        path = Path(args.space_file)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yml", ".yaml"}:
                space = yaml.safe_load(f)
            else:
                space = json.load(f)
    else:
        space = SEARCH_SPACE

    normalized = {}
    for key, values in space.items():
        if not isinstance(values, (list, tuple)):
            values = [values]
        normalized[key] = list(values)
    if not normalized:
        raise ValueError("Search space is empty.")
    return normalized


def set_dotted(config, dotted_key, value):
    node = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_param(config, dataset, key, value):
    if "." in key:
        set_dotted(config, key, value)
        return

    config["model"][key] = value
    dataset_config = config["model"].get(dataset)
    if isinstance(dataset_config, dict):
        dataset_config[key] = value


def apply_params(config, dataset, params):
    for key, value in FIXED_OVERRIDES.items():
        set_dotted(config, key, value)
    for key, value in params.items():
        apply_param(config, dataset, key, value)


def iter_grid(search_space):
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def parse_metric_values(text):
    metrics = {}
    pattern = re.compile(r"'([^']+)'\s*:\s*(?:array\()?\[([^\]]*)\]\)?")
    for metric, raw_values in pattern.findall(text):
        values = []
        for token in re.split(r"[,\s]+", raw_values.strip()):
            if token:
                values.append(float(token))
        metrics[metric] = values
    return metrics


def parse_final_result(output):
    best_epoch = ""
    result_text = ""
    for line in reversed(output.splitlines()):
        if "Final test result:" not in line:
            continue
        match = re.search(r"Best Epoch\s+(\d+)\.\s+Final test result:\s*(.*)", line)
        if match:
            best_epoch = match.group(1)
            result_text = match.group(2).strip()
        else:
            result_text = line.split("Final test result:", 1)[1].strip()
        if result_text.endswith("."):
            result_text = result_text[:-1]
        break
    return best_epoch, result_text, parse_metric_values(result_text)


def metric_columns(config):
    columns = []
    metrics = config.get("test", {}).get("metrics", [])
    ks = config.get("test", {}).get("k", [])
    for metric in metrics:
        for k in ks:
            columns.append("{}@{}".format(metric, k))
    return columns


def flatten_metrics_by_config(metrics, config):
    flat = {}
    ks = config.get("test", {}).get("k", [])
    for metric in config.get("test", {}).get("metrics", []):
        values = metrics.get(metric, [])
        for idx, k in enumerate(ks):
            flat["{}@{}".format(metric, k)] = values[idx] if idx < len(values) else ""
    return flat


def run_command(cmd, cwd):
    output_lines = []
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    return_code = process.wait()
    return return_code, "".join(output_lines)


def write_row(csv_path, fieldnames, row):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    root = repo_root()
    modelconf_dir = root / "encoder" / "config" / "modelconf"
    base_config_path = modelconf_dir / "{}.yml".format(args.model)
    if not base_config_path.exists():
        raise FileNotFoundError(base_config_path)

    base_config = load_yaml(base_config_path)
    search_space = load_search_space(args)
    combinations = list(iter_grid(search_space))
    selected = combinations[args.start_index:]
    if args.max_runs is not None:
        selected = selected[:args.max_runs]

    if args.output:
        csv_path = Path(args.output)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = root / "encoder" / "sweep_results" / "{}_{}_{}.csv".format(args.model, args.dataset, timestamp)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    columns = metric_columns(base_config)
    fieldnames = [
        "run_index",
        "status",
        "elapsed_sec",
        "model",
        "dataset",
        "seed",
        "params",
        "best_epoch",
        "final_result",
        "command",
        "error",
    ] + columns

    print("Search space: {}".format(json.dumps(search_space, ensure_ascii=False)))
    print("Total combinations: {}".format(len(combinations)))
    print("Selected runs: {}".format(len(selected)))
    print("Output CSV: {}".format(csv_path))

    if args.dry_run:
        for offset, params in enumerate(selected):
            run_index = args.start_index + offset
            print("[dry-run] run {} params {}".format(run_index, params))
        return

    for offset, params in enumerate(selected):
        run_index = args.start_index + offset
        temp_model = "_sweep_{}_{}_{}".format(args.model, os.getpid(), run_index)
        temp_config_path = modelconf_dir / "{}.yml".format(temp_model)
        config = load_yaml(base_config_path)
        apply_params(config, args.dataset, params)

        cmd = [
            args.python,
            "encoder/train_encoder.py",
            "--model",
            temp_model,
            "--dataset",
            args.dataset,
            "--device",
            args.device,
            "--cuda",
            str(args.cuda),
        ]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])

        start_time = time.time()
        status = "failed"
        error = ""
        best_epoch = ""
        final_result = ""
        metrics = {}

        print("\n=== Run {}/{} index={} params={} ===".format(offset + 1, len(selected), run_index, params))
        try:
            dump_yaml(temp_config_path, config)
            return_code, output = run_command(cmd, root)
            elapsed = time.time() - start_time
            best_epoch, final_result, metrics = parse_final_result(output)
            if return_code == 0 and final_result:
                status = "ok"
            elif return_code == 0:
                status = "no_final_result"
                error = "Training finished but final result line was not found."
            else:
                status = "failed"
                error = "Process exited with code {}.".format(return_code)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            elapsed = time.time() - start_time
            error = repr(exc)
        finally:
            if temp_config_path.exists():
                temp_config_path.unlink()

        row = {
            "run_index": run_index,
            "status": status,
            "elapsed_sec": "{:.2f}".format(elapsed),
            "model": args.model,
            "dataset": args.dataset,
            "seed": "" if args.seed is None else args.seed,
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "best_epoch": best_epoch,
            "final_result": final_result,
            "command": " ".join(cmd),
            "error": error,
        }
        row.update(flatten_metrics_by_config(metrics, base_config))
        write_row(csv_path, fieldnames, row)
        print("Recorded run {} status={}.".format(run_index, status))


if __name__ == "__main__":
    main()
