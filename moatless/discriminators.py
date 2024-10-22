import logging
from typing import List

from pydantic import BaseModel

from moatless.node import Node

logger = logging.getLogger(__name__)


class Discriminator(BaseModel):
    def select(self, nodes: List[Node]) -> Node | None:
        raise NotImplementedError


class MeanAwardDiscriminator(Discriminator):
    def select(self, nodes: List[Node]) -> Node | None:
        best_finish_node: Node | None = None
        best_mean_reward = float("-inf")
        trajectories_mean_rewards = []

        for finished_node in nodes:
            mean_reward = finished_node.calculate_mean_reward()

            trajectories_mean_rewards.append((finished_node.node_id, mean_reward))
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_finish_node = finished_node

        logger.info(
            f"Mean Rewards for finished trajectories: {trajectories_mean_rewards}"
        )

        if best_finish_node:
            logger.info(
                f"Best finished path finished on Node{best_finish_node.node_id} with mean reward: {best_mean_reward}"
            )
            return best_finish_node
        else:
            logger.info(
                "No valid finished path found. This should not happen if there are finished nodes."
            )
            return None


# TODO: Debate discriminator?
