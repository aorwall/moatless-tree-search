
import logging
from typing import List
from moatless.discriminator.base import BaseDiscriminator
from moatless.node import Node, DiscriminatorResult

logger = logging.getLogger(__name__)

class MeanAwardDiscriminator(BaseDiscriminator):
    
    def select(self, nodes: List[Node]) -> DiscriminatorResult:
        best_finish_node: Node | None = None
        best_mean_reward = float("-inf")
        trajectories_mean_rewards = []
        
        trace = { "mean_rewards": {} }

        for finished_node in nodes:
            mean_reward = finished_node.calculate_mean_reward()

            trajectories_mean_rewards.append((finished_node.node_id, mean_reward))
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_finish_node = finished_node
                
            trace["mean_rewards"][str(finished_node.node_id)] = mean_reward

        logger.info(
            f"Mean Rewards for finished trajectories: {trajectories_mean_rewards}"
        )

        if best_finish_node:
            logger.info(
                f"Best finished path finished on Node{best_finish_node.node_id} with mean reward: {best_mean_reward}"
            )
            return DiscriminatorResult(
                selected_node_id=best_finish_node.node_id,
                trace=trace,
                completion=None
            )
        else:
            logger.info(
                "No valid finished path found. This should not happen if there are finished nodes."
            )
            return DiscriminatorResult(
                selected_node_id=None,
                trace=trace,
                completion=None
            )


class BestRewardDiscriminator(BaseDiscriminator):
    def select(self, nodes: List[Node]) -> DiscriminatorResult:
        best_finish_node: Node | None = None
        
        trace = { "rewards": {} }

        for finished_node in nodes:
            if (
                best_finish_node is None
                or (finished_node.reward is not None 
                    and best_finish_node.reward is not None
                    and finished_node.reward.value > best_finish_node.reward.value)
            ):
                best_finish_node = finished_node

            trace["rewards"][str(finished_node.node_id)] = finished_node.reward.value if finished_node.reward else None

        if best_finish_node:
            logger.info(
                f"Best finished path finished on Node{best_finish_node.node_id} with reward: {best_finish_node.reward.value if best_finish_node.reward else None}"
            )
            return DiscriminatorResult(
                selected_node_id=best_finish_node.node_id,
                trace=trace,
                completion=None
            )
        else:
            logger.info(
                "No valid finished path found. This should not happen if there are finished nodes."
            )
            return DiscriminatorResult(
                selected_node_id=None,
                trace=trace,
                completion=None
            )
