# DiffSES

## Stage I - Neural Policy Learning

> Training a Visual RL agent as a teacher

- Stable baselines 3 for PPO training

Run `train_visual_rl.py` with appropriate environment selected. Refer to stable baselines zoo for additional
configuration options. Use the wrappers provided in `retro_utils.py` for running on retro environments with multiple
lives and stages.

Trained model will be generated in the `logs/` folder (along with tensorboard logs).

## Stage II - Symbolic Fitting

> Distillation of Teacher agent into Symbolic Student

- GPLearn on offline dataset of teacher's actions

### Part A: Training a self-supervised object detector

Training images for multiple atari environments can be
found [here](https://drive.google.com/file/d/1vzFVFhJZDZMkJ8liROtIyzOiUY42r4TZ/view). If you would like to run on
custom/other environments, consider generating them using the provided script `save_frames.py`. We then proceed to train
the OD module using these frames.

For more training parameters, consider referring the scripts and the SPACE project's documentation.

```shell
cd space/
python main.py --task train --config configs/atari_spaceinvaders.yaml resume True device 'cuda:0'
```

This should generate weights in the `space/output/logs` folder. Pretrained models from SPACE are
available [here](https://drive.google.com/file/d/1gUvLTfy5pKeLa6k3RT8GiEXWiGG8XzzD/view).

### Part B: Generating the offline dataset

Save teacher model's behavior (state-action pairs) along with OD module processing all such states. This creates a JSON
of the form. `sample.json` contains a dummy dataset for demonstration purposes.

```
[
  {
    "state": [
      {
        "type": int,
        "x_velocity": float,
        "y_velocity": float,
        "x_position": int,
        "y_position": int
      },
      {
        "type": int,
        "x_velocity": float,
        ...
      }
      ...
    ],
    "teacher_action": int
  }
  ...
]
```

### Part C: Symbolic distillation

We use gplearn's symbolic regression API in `distill_teacher.py` to train a symbolic tree to mimic the teacher's
actions. The operators are as defined in the file and can easily be extended for more operands through the simple
gplearn APIs. Please check `see/judges.py` for a few sample implementations of operators. The operands are the states
from JSON as stored. We recommend running this experiment numerous times to achieve good performance as convergence of
such a random search is not a guarantee every time. Please refer to `gplearn_optuna.py` for a sample of automating such
a search on random data.

## Stage III - Fine-tuning Symbolic Tree

> Neural Guided Differentiable Search

Lastly, our symbolic finetuning stage consists of `symbolic_finetuning.py` which uses a custom implementation of gplearn
modified in order to support the following:

- **RL style training:** rewards as a fitness metric rather than MSE with respect to teacher behavior.
- **Differentiable constant optimization:** new mutation scheme where the constants are set to be differentiable, the
  tree acts as the policy network for a PPO agent and optimization is performed on those constants.
- **Soft expert supervision in loss:** add-on to earlier bullet along with an extra loss term to aforementioned loss
  being
  the difference between the teacher's action and the symbolic tree's prediction.

While running that file, please run a `pip install -e .` inside the custom implementation of gplearn to install the
local version instead of the prebuilt wheels from PyPi. Similar to [Part 2.3](#part-c--symbolic-distillation), we
recommend running this experiment numerous times to achieve
acceptable levels of convergence.