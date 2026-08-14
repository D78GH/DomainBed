# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.visualizations import prepare_prototype_pca, plot_prototypes, plot_domain_generalization, plot_prototype_utilisation, plot_prototype_class_heatmap, plot_prototype_mutual_information, plot_learning_dynamics, compute_prototype_mi # JP added

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    print("N_WORKERS =", dataset.N_WORKERS)

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        # num_workers=dataset.N_WORKERS)
        num_workers=0)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    print("Train loaders created")

    print("Fetching first batch...")
    batch = next(iter(train_loaders[0]))
    print("First batch fetched")

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    # JP added: collect a small fixed set of samples for visualisation.
    def collect_visualization_data(split, max_samples=1000):
        loader = FastDataLoader(dataset=split, batch_size=64, num_workers=0)
        xs = []
        ys = []
        total = 0
        for x, y in loader:
            remaining = max_samples - total
            if remaining <= 0:
                break
            x = x[:remaining]
            y = y[:remaining]
            xs.append(x)
            ys.append(y)
            total += len(x)
        if not xs:
            raise ValueError("No samples available for visualisation.")
        return torch.cat(xs), torch.cat(ys)

    # JP added: prepare training-domain and unseen-domain visualisation data.
    train_env_indices = [i for i in range(len(dataset)) if i not in args.test_envs]
    if len(train_env_indices) == 0:
        raise ValueError("No training environments available for visualisation.")

    train_vis_x, train_vis_y = collect_visualization_data(
        in_splits[train_env_indices[0]][0], 1000)

    unseen_vis_x, unseen_vis_y = collect_visualization_data(
        out_splits[args.test_envs[0]][0], 1000)

    visualization_x = torch.cat([train_vis_x, unseen_vis_x], dim=0)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([
        len(env) / hparams['batch_size']
        for env, _ in in_splits
    ])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    # JP added: store loss values for the learning-dynamics visualisation.
    learning_history = []

    last_results_keys = None

    for step in range(start_step, n_steps):
        print("STEP START", step)
        step_start_time = time.time()

        print("Getting minibatch...")
        minibatches_device = [
            (x.to(device), y.to(device))
            for x, y in next(train_minibatches_iterator)
        ]
        print("Minibatch moved to device")

        if args.task == "domain_adaptation":
            uda_device = [
                x.to(device)
                for x, _ in next(uda_minibatches_iterator)
            ]
        else:
            uda_device = None

        print("Calling algorithm.update...")
        step_vals = algorithm.update(minibatches_device, uda_device)
        print("algorithm.update finished")

        checkpoint_vals['step_time'].append(
            time.time() - step_start_time
        )

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        # JP added: create visualisations at selected training checkpoints.
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            visualisation_steps={0,1000,2500,5000,n_steps-1}

            if step in visualisation_steps:
                print(f"Creating visualisations for step {step}...")

                visualization_pca = prepare_prototype_pca(
                    algorithm,
                    visualization_x,
                    max_samples=500,
                    batch_size=16
                )

                plot_prototypes(
                    algorithm,
                    visualization_x,
                    torch.cat([train_vis_y, unseen_vis_y]),
                    visualization_pca,
                    step=step,
                    max_samples=500,
                    batch_size=16,
                    output_dir=args.output_dir,
                    test_env=args.test_envs[0]
                )

                plot_domain_generalization(
                    algorithm,
                    train_vis_x,
                    train_vis_y,
                    unseen_vis_x,
                    unseen_vis_y,
                    visualization_pca,
                    step=step,
                    max_samples=500,
                    batch_size=16,
                    output_dir=args.output_dir,
                    test_env=args.test_envs[0]
                )

                plot_prototype_utilisation(
                    algorithm,
                    train_vis_x,
                    train_vis_y,
                    step=step,
                    max_samples=1000,
                    batch_size=16,
                    output_dir=args.output_dir,
                    test_env=args.test_envs[0]
                )

                plot_prototype_class_heatmap(
                    algorithm,
                    train_vis_x,
                    train_vis_y,
                    step=step,
                    max_samples=1000,
                    batch_size=16,
                    output_dir=args.output_dir,
                    test_env=args.test_envs[0]
                )

                print(f"Finished visualisations for step {step}")

            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # JP added: record averaged losses for learning-dynamics visualisation.
            prototype_mi = compute_prototype_mi(
                algorithm,
                train_vis_x,
                train_vis_y,
                batch_size=16
            )

            results["prototype_mi"] = prototype_mi

            learning_history.append({
                "step": step,
                "loss": results.get("loss"),
                "ce_loss": results.get("ce_loss"),
                "proto_loss": results.get("proto_loss"),
                "mem_loss": results.get("mem_loss"),
                "prototype_mi": prototype_mi
            })

            print("Starting evaluation")
            eval_start = time.time()

            evals = zip(eval_loader_names, eval_loaders, eval_weights)

            for name, loader, weights in evals:
                print(f"Evaluating {name}...")
                env_start = time.time()

                acc = misc.accuracy(
                    algorithm,
                    loader,
                    weights,
                    device
                )

                env_time = time.time() - env_start

                print(
                    f"Finished {name}: "
                    f"acc={acc:.4f}, "
                    f"time={env_time:.2f}s"
                )

                results[name + '_acc'] = acc

            total_eval_time = time.time() - eval_start

            print(
                f"Finished evaluation. "
                f"Total evaluation time: {total_eval_time:.2f}s"
            )

            if torch.cuda.is_available():
                results['mem_gb'] = (
                    torch.cuda.max_memory_allocated()
                    / (1024. * 1024. * 1024.)
                )
            else:
                results['mem_gb'] = 0.0

            results_keys = sorted(results.keys())

            if results_keys != last_results_keys:
                misc.print_row(
                    results_keys,
                    colwidth=12
                )
                last_results_keys = results_keys

            misc.print_row(
                [results[key] for key in results_keys],
                colwidth=12
            )

            results.update({
                'hparams': hparams,
                'args': vars(args),
                'hparams_seed': args.hparams_seed,
                'seed': args.seed,
                'trial_seed': args.trial_seed,
            })

            for k, v in results.items():
                try:
                    json.dumps(v)
                except TypeError:
                    print(f"Non-serializable field: {k}")
                    print(f"Type: {type(v)}")
                    print(f"Value: {v}")

            epochs_path = os.path.join(
                args.output_dir,
                'results.jsonl'
            )

            with open(epochs_path, 'a') as f:
                f.write(
                    json.dumps(
                        results,
                        sort_keys=True
                    ) + "\n"
                )

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(
                    f'model_step{step}.pkl'
                )

    # JP added: save the learning-dynamics and prototype MI visualisations after training.
    if learning_history:
        plot_learning_dynamics(
            learning_history,
            args.output_dir,
            test_env=args.test_envs[0]
        )
        plot_prototype_mutual_information(
            learning_history,
            args.output_dir,
            test_env=args.test_envs[0]
        )

    save_checkpoint('model.pkl')

    with open(
        os.path.join(args.output_dir, 'done'),
        'w'
    ) as f:
        f.write('done')