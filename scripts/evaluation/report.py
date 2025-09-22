"""Prepare ECHI data"""

import csv
import itertools
import json
import logging
import os
from glob import glob

import hydra
import numpy as np
from omegaconf import DictConfig

from evaluation.segment_signals import get_session_tuples

EARLIEST_START = 30 * 16000  # 30 seconds in sample
NON_NUMERIC_KEYS = {"key", "srmr"}


def read_jsonl(file_path, data=None):
    """Read a JSONL file and return a list of dictionaries."""
    if data is None:
        data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def parse_name(name):
    """Parse name to extract start and end time stamps in samples.

    Segment names have the format: dev_05.aria.P123.092.13174243_13314467.wav
    where the last part is the start and end time stamps in samples
    """
    return name.split(".")[-2].split("_")


def compute_stats(results):
    """Compute statistics from the results."""
    stats = dict()
    valid_keys = [k for k in results[0] if k not in NON_NUMERIC_KEYS]

    for key in valid_keys:
        durations = [result["duration"] for result in results if key in result]
        data = [float(result[key]) for result in results if key in result]
        weighted_mean = np.average(data, weights=durations) if data else np.nan
        mean_value = np.mean(data) if data else np.nan
        std_value = np.std(data) if len(data) > 1 else np.nan
        min_value = np.min(data) if data else np.nan
        max_value = np.max(data) if data else np.nan
        stats[key] = {
            "weighted_mean": weighted_mean,
            "mean": mean_value,
            "std": std_value,
            "std_err": std_value / np.sqrt(len(data)) if len(data) > 1 else np.nan,
            "count": len(data),
            "min": min_value,
            "max": max_value,
        }
    return stats


def save_stats(stats, stats_file):
    """Save computed statistics to a file."""
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    logging.info(f"Stats saved to {stats_file}")


def save_results(results, results_file, ext=None):
    """Save results to a CSV file."""
    if ext is not None:
        results_file_base, _ = os.path.splitext(results_file)
        results_file = f"{results_file_base}{ext}"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def preprocess_results(raw_results):
    """Preprocess results.DictConfig
    - Remove empty segments
    - Remove segments that start before the end of the 30 second intro
    - Add duration and start time fields
    """
    results = []
    for result in raw_results:
        if len(result.keys()) == 1:
            logging.warning(f"Segment {result['key']} how no results")
            continue
        results.append(result)

    for result in results:
        start_time, end_time = parse_name(result["key"])
        result["duration"] = int(end_time) - int(start_time)
        result["start_time"] = int(start_time)

    results = [result for result in results if result["start_time"] >= EARLIEST_START]

    return results


def report(cfg):
    logging.info("Reporting results")

    session_device_pid_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=[cfg.dataset]
    )

    sessions = set(session for session, _, _ in session_device_pid_tuples)
    pids = set(pid for _, _, pid in session_device_pid_tuples)

    for device, segment_type in itertools.product(cfg.devices, cfg.segment_types):
        logging.info(f"Processing device: {device}")
        results_file = cfg.results_file.format(
            dataset=cfg.dataset, device=device, segment_type=segment_type
        )
        results_file_base, ext = os.path.splitext(results_file)
        # The wildcard is used to accumulate over multiple batches
        results_files = glob(f"{results_file_base}*{ext}")

        if not results_files:
            logging.warning(f"No results found for device: {device}")
            continue

        results = []
        for results_file in results_files:
            results = read_jsonl(results_file, results)
        logging.info(f"Total results for {device}: {len(results)}")

        results = preprocess_results(results)

        stats = compute_stats(results)
        stats_file = cfg.report_file.format(
            dataset=cfg.dataset,
            device=device,
            segment_type=segment_type,
            session="_",
            pid="_",
        )
        save_stats(stats, stats_file)
        save_results(results, stats_file, ext=".csv")

        # Process the session level reports for each device
        for session in sessions:
            session_results = [result for result in results if session in result["key"]]
            if not session_results:
                logging.warning(f"No results for session: {session}")
                continue

            session_stats = compute_stats(session_results)
            session_stats_file = cfg.report_file.format(
                dataset=cfg.dataset,
                device=device,
                segment_type=segment_type,
                session=session,
                pid="_",
            )
            save_stats(session_stats, session_stats_file)
            save_results(session_results, session_stats_file, ext=".csv")

            # Process the PID level reports for each device-session combination
            for pid in pids:
                pid_session_results = [
                    result for result in session_results if pid in result["key"]
                ]
                if not pid_session_results:
                    continue

                participant_stats = compute_stats(pid_session_results)
                participant_stats_file = cfg.report_file.format(
                    dataset=cfg.dataset,
                    device=device,
                    segment_type=segment_type,
                    session=session,
                    pid=pid,
                )
                save_stats(participant_stats, participant_stats_file)
                save_results(pid_session_results, participant_stats_file, ext=".csv")


@hydra.main(
    version_base=None, config_path="../../config/evaluation", config_name="main"
)
def main(cfg: DictConfig) -> None:
    report(cfg.report)


if __name__ == "__main__":
    main()
