import json

import numpy as np
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

raw_data = json.load(open("sample.json"))

# find different entity types
object_types = set(eachh["type"] for each in raw_data for eachh in each["state"])
# mac times they occur in a single state
max_times = [0] * (max(object_types) + 1)
for timestep in raw_data:
    times = [0] * (max(object_types) + 1)
    for entity in timestep["state"]:
        times[entity["type"]] += 1
    max_times = np.max(np.stack([max_times, times]), 1)

# create offline dataset with a zeros buffer for "max_appearances - num_appearances" for each entity
x_train = []
for timestep in raw_data:
    x_timestep_dict = {each: [] for each in object_types}
    for entity in timestep["state"]:
        x_timestep_dict[entity["type"]].append(entity["x_velocity"])
        x_timestep_dict[entity["type"]].append(entity["y_velocity"])
        x_timestep_dict[entity["type"]].append(entity["x_position"])
        x_timestep_dict[entity["type"]].append(entity["y_position"])
    x_timestep_list = []
    for (object_type, values), max_time in zip(x_timestep_dict.items(), max_times):
        x_timestep_list.append(values + [0.0] * (max_time * 4 - len(values)))
    x_train.append(sum(x_timestep_list, []))

# convert to numpy arrays
x_train = np.array(x_train)  # none, num_features
y_train = np.array([each["teacher_action"] for each in raw_data])  # none


# initialize custom operators similar to judges/see.py
def _logical(x1, x2):
    return np.where(x1 > x2)


def _is_right_of(self, other):
    condition = other.position[0] - self.position[0]
    return condition < 0


def _is_left_of(self, other):
    condition = other.position[0] - self.position[0]
    return condition > 0


def _is_top_of(self, other):
    condition = other.position[1] - self.position[1]
    return condition > 0


def _is_bottom_of(self, other):
    condition = other.position[1] - self.position[1]
    return condition < 0


logical = make_function(function=_logical, name="logical", arity=2)
is_right_of = make_function(function=_is_right_of, name="is_right_of", arity=2)
is_left_of = make_function(function=_is_left_of, name="is_left_of", arity=2)
is_top_of = make_function(function=_is_top_of, name="is_top_of", arity=2)
is_bottom_of = make_function(function=_is_bottom_of, name="is_bottom_of", arity=2)

# run symbolic regression on offline data
est_gp = SymbolicRegressor(
    verbose=1,
    n_jobs=4,
    function_set=(
        "add",
        "sub",
        "mul",
        logical,
        # is_right_of,
        # is_left_of,
        # is_top_of,
        # is_bottom_of,
        # add more as required
    ),
    population_size=1000,
    generations=100,
    tournament_size=25,
    stopping_criteria=0.0,
    const_range=(-3.0, 3.0),
    init_depth=(2, 6),
    init_method="half and half",
    metric="mean absolute error",
    parsimony_coefficient=0.001,
    p_crossover=0.9,
    p_subtree_mutation=0.01,
    p_hoist_mutation=0.01,
    p_point_mutation=0.01,
    p_point_replace=0.05,
    max_samples=1.0,
)
est_gp.fit(x_train, y_train)
print(est_gp._program)
