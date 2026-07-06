# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
import pandas as pd # JP added
EXCEL_ROWS = []

def remove_key(d,key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def recursive_freeze(obj):
    if isinstance(obj, dict):
        return frozenset((key, recursive_freeze(val)) for key, val in obj.items())
    elif isinstance(obj, list):
        return tuple(recursive_freeze(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(recursive_freeze(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(recursive_freeze(item) for item in obj)
    else:
        return obj

def merge_records(records):
    merged_records = []
    args_set = set()  # Store unique args dictionaries

    # Group records by unique 'args' dictionaries
    for record in records:
        args = record['args'].copy()
        args.pop('holdout_fraction', None)  # Remove 'holdout_fraction' from comparison
        args_key = recursive_freeze(args)
        args_set.add(args_key)

    # Merge records with the same 'args' except for 'holdout_fraction'
    for args_key in args_set:
        args_dict = dict(args_key)
        filtered_records = [record for record in records if dict(recursive_freeze(remove_key(record['args'],'holdout_fraction'))) == args_dict]
        merged_record = {}
        for record in filtered_records:
            merged_record.update(record)
        merged_records.append(merged_record)
    return Q(merged_records)

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error JP replaced function"""
    vals = [x for x in data if x is not None]

    if len(vals) == 0:
        return None, None, "X"

    mean = 100 * np.mean(vals)
    err = 100 * np.std(vals) / np.sqrt(len(vals))

    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""

    grouped_records = reporting.get_grouped_records(records)

    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
                 [n for n in alg_names if n not in algorithms.ALGORITHMS])

    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    if selection_method == model_selection.IIDAutoLRAccuracySelectionMethod:
        for r in grouped_records:
            r['records'] = merge_records(r['records'])

    grouped_records = grouped_records.map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    print(dataset_names)
    print(alg_names)
    print(len(grouped_records))

  # JP added: Excel logging (simplified)
    global EXCEL_ROWS

    for dataset in dataset_names:
        test_envs = range(datasets.num_environments(dataset))

        for algorithm in alg_names:
            for test_env in test_envs:

                trial_accs = (
                    grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    )
                    .select("sweep_acc")
                )

                vals = [x for x in trial_accs if x is not None]
                if len(vals) == 0:
                    continue

                acc = float(np.mean(vals)) * 100
                std = float(np.std(vals)) * 100

                n_trials = len(vals)

                domains = datasets.get_dataset_class(dataset).ENVIRONMENTS

                heldout_domain = domains[test_env]

                # matching = grouped_records.filter_equals(
                #     "dataset, algorithm, test_env",
                #     (dataset, algorithm, test_env)
                # )

                # hparams_seed = None
                # seed = None

                # if len(matching):
                #     first_record = matching[0]["records"][0]
                #     trial_seed = first_record.get("trial_seed")
                #     hparams_seed = first_record.get("hparams_seed")
                #     seed = first_record.get("seed")

                EXCEL_ROWS.append({
                    "Dataset": dataset,
                    "Model": algorithm,
                    "Held-out Domain": heldout_domain,
                    "Selection Method": selection_method.name,
                    # "Trial Seed": trial_seed,
                    # "HParams Seed": hparams_seed,
                    # "Seed": seed,
                    "N Trials": n_trials,
                    "Standard Deviation": std,
                    "Mean Accuracy": acc,
                })

    # Print dataset tables
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

    # Averages table
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
                     group.select("sweep_acc").mean())
            )

            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)

        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"

    print_table(table, header_text, alg_names, col_labels,
                colwidth=25, latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--auto_lr", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    if args.auto_lr:
        SELECTION_METHODS = [model_selection.IIDAutoLRAccuracySelectionMethod]
    else:
        SELECTION_METHODS = [
            model_selection.IIDAccuracySelectionMethod,
            model_selection.LeaveOneOutSelectionMethod,
            model_selection.OracleSelectionMethod,
        ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex)
    
    # JP added
    if len(EXCEL_ROWS) > 0:
        out_path = os.path.join(args.input_dir, "domainbed_results.xlsx")

        df = pd.DataFrame(EXCEL_ROWS)

        # Pivot: selection methods become columns
        df_pivot = df.pivot_table(
            index=[
                "Dataset",
                "Model",
                "Held-out Domain",
                "N Trials"
            ],
            columns="Selection Method",
            values=["Mean Accuracy", "Standard Deviation"]
        )

        # flatten MultiIndex columns
        df_pivot = df_pivot.reset_index()

        df_pivot.columns = [
            "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
            for col in df_pivot.columns
        ]

        # merge with old file (same logic you already had)
        if os.path.exists(out_path):
            old = pd.read_excel(out_path)
            df_pivot = pd.concat([old, df_pivot], ignore_index=True)

            df_pivot = df_pivot.drop_duplicates(
                subset=[
                    "Dataset",
                    "Model",
                    "Held-out Domain",
                    "N Trials"
                ]
            )

        df_pivot.to_excel(out_path, index=False)

        print(f"\n[Excel saved] {out_path}")

    if args.latex:
        print("\\end{document}")
