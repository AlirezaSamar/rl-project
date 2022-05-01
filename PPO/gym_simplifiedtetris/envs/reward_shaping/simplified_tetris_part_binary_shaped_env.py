"""Contains a simplified Tetris env with a part-binary obs space and shaping reward function."""

from gym_simplifiedtetris.register import register_env
from gym_simplifiedtetris.envs.simplified_tetris_part_binary_env import (
    SimplifiedTetrisPartBinaryEnv,
)
from ._potential_based_shaping_reward import _PotentialBasedShapingReward


class SimplifiedTetrisPartBinaryShapedEnv(
    _PotentialBasedShapingReward, SimplifiedTetrisPartBinaryEnv
):
    """
    A simplified Tetris env, where the reward function is a
    scaled heuristic score and the obs space is the grid's part binary
    representation plus the current piece's id.

    :param grid_dims: the grid's dimensions.
    :param piece_size: the size of the pieces in use.
    :param seed: the rng seed.
    """

    def __init__(self, **kwargs):
        """Extend the two superclasses."""
        super().__init__()
        SimplifiedTetrisPartBinaryEnv.__init__(self, **kwargs)


register_env(
    incomplete_id="simplifiedtetris-partbinary-shaped",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryShapedEnv",
)
