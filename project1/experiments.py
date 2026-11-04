from collections import OrderedDict
from pathlib import Path
import part1.experiment
from project1 import models


def mlp2_cross_validation(data_splits, outcome_space, encoder, save_to, mlp2_args=dict()):
    """
    cross_validation_experiment for our two-hidden-layer MLP, with the same return format
    """
    results = {'models': OrderedDict(), 'predictions': OrderedDict(), 'accuracies': OrderedDict()}

    for fold, split in data_splits.items():
        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True)

        metrics, best_epoch, predictions = models.train_mlp2(
            split["training"], split["validation"], len(outcome_space), encoder,
            save_to=f"{fold_folder}/model", **mlp2_args
        )
        results['models'][fold] = f"{fold_folder}/model"
        results['predictions'][fold] = predictions
        results['accuracies'][fold] = metrics[best_epoch]['val_metrics']['accuracy']

    return results


def run_experiments(representations, architectures, pairs, outcome_space,
        results_dir="../results", seeds=(0,), num_folds=5):
    """
    runs experiments on all combinations of representations and architectures
    """
    my_classifiers = OrderedDict()

    for rep_name, encoder in representations.items():
        for arch_name, config in architectures.items():
            key = f"{arch_name}_{rep_name}"
            runs = []
            for seed in seeds:
                cv = part1.experiment.cross_validation_split(pairs, k=num_folds, seed=seed)
                if config["model_option"] == "mlp2":
                    result = mlp2_cross_validation(
                        cv, outcome_space, encoder,
                        save_to=f"{results_dir}/{key}/seed-{seed}",
                        mlp2_args=config.get("mlp2_args", {})
                    )
                else:
                    result = part1.experiment.cross_validation_experiment(
                        data_splits=cv,
                        outcome_space=outcome_space,
                        encoder=encoder,
                        model_option=config["model_option"],
                        save_to=f"{results_dir}/{key}/seed-{seed}",
                        # gets args if they exist otherwise empty dict
                        loglin_args=config.get("loglin_args", {}),
                        smlp_args=config.get("smlp_args", {})
                    )
                runs.append(result)
            # runs needs to be a list for plotting
            my_classifiers[key] = runs

    return my_classifiers




