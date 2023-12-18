# I came up with a cool idea (for once)

Rewards come from the reward function as an array. The learner is told how to use these with a list
of tuples that look like `(doScaling: bool, min: float, target: float, max: float)`

`doScaling` will toggle whether scaling logic is applied.

`min` is the minimum value the reward can reach.

`target` is the reward value that the adaptive scaling will target.

`max` is the maximum value the reward can reach.

## Code changes

- Implement the adaptive scaling stuff

- Log scale coefficient and scaled value for each reward

- Create ZipReward, which just zips other rewards together.

## Experiment design

- use fairly default values from example.py

- Will need to add a new reward or two to make a good comparison, not sure what that looks like.
