"""gnn_hyperopt.py

Hyperopt parameters for FFN model

"""
import os
import copy
import logging
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ray import tune

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.massformer_pred.train as massformer_train
import ms_pred.massformer_pred.massformer_data as massformer_data
import ms_pred.massformer_pred.massformer_model as massformer_model


def score_function(config, base_args, orig_dir=""):
    """score_function.

    Args:
        config: All configs passed by hyperoptimizer
        base_args: Base arguments
        orig_dir: ""
    """
    # tunedir = tune.get_trial_dir()
    # Switch s.t. we can use relative data structures
    os.chdir(orig_dir)

    kwargs = copy.deepcopy(base_args)
    kwargs.update(config)
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values

    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    num_bins = kwargs.get("num_bins")
    num_workers = kwargs.get("num_workers", 0)
    upper_limit = kwargs.get("upper_limit", 1500)

    train_dataset = massformer_data.BinnedDataset(
        train_df,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        form_dir_name=kwargs["form_dir_name"],
    )
    val_dataset = massformer_data.BinnedDataset(
        val_df,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        form_dir_name=kwargs["form_dir_name"],
    )

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Define model
    # test_batch = next(iter(train_loader))
    logging.info("Building model")
    model = massformer_model.MassFormer(
        mf_num_ff_num_layers=kwargs["mf_num_ff_num_layers"],
        mf_ff_h_dim=kwargs["mf_ff_h_dim"],
        mf_ff_skip=kwargs["mf_ff_skip"],
        mf_layer_type=kwargs["mf_layer_type"],
        mf_dropout=kwargs["mf_dropout"],
        use_reverse=kwargs["use_reverse"],
        embed_adduct=kwargs['embed_adduct'],

        gf_model_name=kwargs["gf_model_name"],
        gf_pretrain_name=kwargs["gf_pretrain_name"],
        gf_fix_num_pt_layers=kwargs["gf_fix_num_pt_layers"],
        gf_reinit_num_pt_layers=kwargs["gf_reinit_num_pt_layers"],
        gf_reinit_layernorm=kwargs["gf_reinit_layernorm"],

        learning_rate=kwargs["learning_rate"],
        lr_decay_rate=kwargs["lr_decay_rate"],
        output_dim=num_bins,
        upper_limit=upper_limit,
        weight_decay=kwargs["weight_decay"],
        loss_fn=kwargs["loss_fn"],
    )

    # outputs = model(test_batch['fps'])
    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(tune.get_trial_dir(), "", ".")

    # Replace with custom callback that utilizes maximum loss during train
    tune_callback = nn_utils.TuneReportCallback(["val_loss"])

    val_check_interval = None
    check_val_every_n_epoch = 1
    monitor = "val_loss"

    # tb_path = tb_logger.log_dir
    earlystop_callback = EarlyStopping(monitor=monitor, patience=10)
    callbacks = [earlystop_callback, tune_callback]
    logging.info("Starting train")
    trainer = pl.Trainer(
        logger=[tb_logger],
        accelerator="gpu" if kwargs["gpu"] else None,
        devices=1 if kwargs["gpu"] else None,
        callbacks=callbacks,
        gradient_clip_val=5,
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)


def get_args():
    parser = argparse.ArgumentParser()
    massformer_train.add_massformer_train_args(parser)
    nn_utils.add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial):
    """get_param_space.

    Use optuna to define this dynamically

    """
    trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    trial.suggest_float("lr_decay_rate", 0.7, 1.0, log=True)
    trial.suggest_categorical("weight_decay", [1e-6, 1e-7, 0])
    trial.suggest_categorical("batch_size", [32, 64, 128])

    # Replace everything above with a suggest categorical
    trial.suggest_categorical("gf_fix_num_pt_layers", [0])
    trial.suggest_categorical("gf_reinit_num_pt_layers", [-1])
    trial.suggest_categorical("gf_reinit_layernorm", [True])
    trial.suggest_categorical("gf_model_name", ["graphormer_base"])
    trial.suggest_categorical("gf_pretrain_name", ["pcqm4mv2_graphormer_base"])
    trial.suggest_categorical("embed_adduct", [True])
    trial.suggest_categorical("use_reverse", [True])

    # Replace
    trial.suggest_categorical("mf_ff_skip", [True])
    trial.suggest_categorical("mf_ff_layer_type", ["neims"])

    # Tuned params
    trial.suggest_float("mf_dropout", 0, 0.5, step=0.1)
    trial.suggest_categorical("mf_ff_h_dim", [64, 128, 256, 512, 1024])
    trial.suggest_int("mf_num_ff_num_layers", 1, 5)



def get_initial_points() -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    init_base = {
        "learning_rate": 0.00086,
        "lr_decay_rate": 0.95,
        "batch_size": 64,
        "weight_decay": 0,
        "dropout": 0.2,

        "mf_ff_h_dim": 256,
        "mf_num_ff_num_layers": 3,
        "mf_dropout": 0.2,
        "mf_ff_skip": True,
        "mf_layer_type": "neims",


        # Define defaults from above
        "embed_adduct": True,
        "use_reverse": True,
        "gf_fix_num_pt_layers": 0,
        "gf_reinit_num_pt_layers": -1,
        "gf_reinit_layernorm": True,
        "gf_model_name": "graphormer_base",
        "gf_pretrain_name": "pcqm4mv2_graphormer_base",
    }
    return [init_base]


def run_hyperopt():
    args = get_args()
    kwargs = args.__dict__
    nn_utils.run_hyperopt(
        kwargs=kwargs,
        score_function=score_function,
        param_space_function=get_param_space,
        initial_points=get_initial_points(),
    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_hyperopt()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
