import numpy as np
np.random.seed(30230433)

from simulation import Simulation


# todo: goals
#[DONE] - add on screen statistics6
#[DONE] - display desired image on screen
#[DONE] - display best image on screen
# - add approprate stats below each one
# - visulisation for best weights
# - add total time elapsed
# - add session time elapsed
# - sort out font sizes
# - sort out font
#[DONE] - pause and play simulation
#[DONE] - remove titlebar
# - make window larger
# - add seconds remaining to frame time left
#[DONE] - add an area to show current best result
#[DONE] - add an area showing the difference needed

# - add statistics to see how many sims are worse and how many are better


if __name__ == '__main__':
    Simulation().run() 