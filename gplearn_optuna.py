import os
import pickle

import joblib
import numpy as np
import optuna
from gplearn.genetic import SymbolicRegressor


class Objective:
    def __init__(self, inputs, outputs):
        # Dataset
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, trial: optuna.Trial):
        # If required, please uncomment hyperparameters and include the kwargs appropriately in est_gp
        # population_size = trial.suggest_int("population_size", 1000, 2000, step=50)
        # tournament_size = trial.suggest_int("tournament_size", 20, 40, step=5)
        # init_depth_min = trial.suggest_int("init_depth_min", 2, 8)
        # init_depth_max = init_depth_min + trial.suggest_int("init_depth_diff", 0, 6)
        # population_size = 1000
        # tournament_size = 20
        # init_depth_min = 2
        # init_depth_max = 6
        # parsimony_coefficient = trial.suggest_float("parsimony_coefficient", 0.0001, 0.1)

        directory_name = f"outputs/{trial.study.study_name}"
        model_name = f"{trial.number}.pkl"
        os.makedirs(directory_name, exist_ok=True)

        est_gp = SymbolicRegressor(
            verbose=0,
            n_jobs=4,
            function_set=("add", "sub", "mul"),
        )
        est_gp.fit(self.inputs, self.outputs)

        with open(os.path.join(directory_name, model_name), "wb") as f:
            pickle.dump(est_gp, f)

        return est_gp.run_details_["best_fitness"][-1]


def main():
    x0 = np.random.rand(10024, 1)
    x1 = np.random.rand(10024, 1)
    x2 = np.random.rand(10024, 1)
    y0 = x0 + x1 - x2

    x_train = np.concatenate([x0, x1, x2], 1)
    y_train = y0.squeeze()

    study_name = "distill-offline-dataset-v1"
    directory_name = f"outputs/{study_name}"
    os.makedirs(directory_name, exist_ok=True)
    study = optuna.create_study(
        storage=f"sqlite:///{directory_name}/{study_name}.db",
        study_name=study_name,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(Objective(x_train, y_train))
    joblib.dump(study, f"{directory_name}/study.pkl")


if __name__ == "__main__":
    main()
