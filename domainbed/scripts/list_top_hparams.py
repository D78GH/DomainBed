# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Example usage:
python -u -m domainbed.scripts.list_top_hparams \
    --input_dir domainbed/misc/test_sweep_data --algorithm ERM \
    --dataset VLCS --test_env 0
"""

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

def compare_hparams(hparams1, hparams2, name1="config 1", name2="config 2"):
    """
    Print only hyperparameters that differ between two configurations.
    """
    print("\n=== HYPERPARAMETER DIFFERENCES ===")

    keys = sorted(set(hparams1.keys()) | set(hparams2.keys()))

    differences = False

    for k in keys:
        v1 = hparams1.get(k)
        v2 = hparams2.get(k)

        if v1 != v2:
            differences = True
            print(f"{k}:")
            print(f"  {name1}: {v1}")
            print(f"  {name2}: {v2}")

    if not differences:
        print("No differences found.")

def todo_rename(records, selection_method, latex):

    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm", 
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--test_env', type=int, required=True)
    args = parser.parse_args()

    records = reporting.load_records(args.input_dir)
    print("Total records:", len(records))

    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r:
            r['dataset'] == args.dataset and
            r['algorithm'] == args.algorithm and
            r['test_env'] == args.test_env
    )

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        model_selection.LeaveOneOutSelectionMethod,
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        print(f'Model selection: {selection_method.name}')

        for group in records:
            print(f"trial_seed: {group['trial_seed']}")

            best_hparams = selection_method.hparams_accs(group['records'])

            if len(best_hparams) == 0:
                print("\tNo valid hyperparameter selection for this method")
                print("-" * 60)
                continue

            # Best hyperparameter configuration
            best_val_acc, best_records = best_hparams[0]
            best_record = best_records[0]

            test_env = group["test_env"]
            best_test_acc = best_record[f"env{test_env}_out_acc"]

            print("\n\tBEST HYPERPARAMETERS")
            print(f"\tvalidation_acc: {best_val_acc}")
            print(f"\ttest_env: {test_env}")
            print(f"\ttest_accuracy: {best_test_acc}")

            print("\thparams:")
            for k, v in sorted(best_record["hparams"].items()):
                print(f"\t\t{k}: {v}")

            print("\toutput_dirs:")
            output_dirs = best_records.select("args.output_dir").unique()
            for output_dir in output_dirs:
                print(f"\t\t{output_dir}")


            # Compare against second-best configuration if available
            if len(best_hparams) > 1:

                second_val_acc, second_records = best_hparams[1]
                second_record = second_records[0]

                second_test_acc = second_record[f"env{test_env}_out_acc"]

                print("\n\tSECOND BEST")
                print(f"\tvalidation_acc: {second_val_acc}")
                print(f"\ttest_accuracy: {second_test_acc}")

                print("\n\t=== HYPERPARAMETER DIFFERENCES ===")

                best_hp = best_record["hparams"]
                second_hp = second_record["hparams"]

                keys = sorted(set(best_hp.keys()) | set(second_hp.keys()))

                differences = False

                for k in keys:
                    best_value = best_hp.get(k)
                    second_value = second_hp.get(k)

                    if best_value != second_value:
                        differences = True
                        print(f"\t{k}:")
                        print(f"\t\tbest:        {best_value}")
                        print(f"\t\tsecond best: {second_value}")

                if not differences:
                    print("\tNo hyperparameter differences")

            print("\n" + "-" * 60)

                # best_hparams = selection_method.hparams_accs(group['records'])
                # for run_acc, hparam_records in best_hparams:
                #     print(f"\t{run_acc}")
                #     for r in hparam_records:
                #         assert(r['hparams'] == hparam_records[0]['hparams'])
                #     print("\t\thparams:")
                #     for k, v in sorted(hparam_records[0]['hparams'].items()):
                #         print('\t\t\t{}: {}'.format(k, v))
                #     print("\t\toutput_dirs:")
                #     output_dirs = hparam_records.select('args.output_dir').unique()
                #     for output_dir in output_dirs:
                #         print(f"\t\t\t{output_dir}")