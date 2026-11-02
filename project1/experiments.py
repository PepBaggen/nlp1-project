from collections import OrderedDict
import part1.experiment


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




