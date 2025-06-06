import logging
from typing import Optional, Tuple

from moatless.actions.search_base import SearchBaseArgs
from moatless.node import Node, ActionStep
from moatless.value_function.base import BaseValueFunction
from moatless.node import Reward

logger = logging.getLogger(__name__)


class FailReasonsValueFunction(BaseValueFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def get_reward(self, node: Node) -> Optional[Reward]:
        fail_reasons = []
        for action_step in node.action_steps:
            if action_step.observation and action_step.observation.properties and action_step.observation.properties.get("fail_reason"):
                fail_reasons.append(action_step.observation.properties.get("fail_reason"))

        if fail_reasons:
            reward_value = max(-100, len(fail_reasons) * -25)
            return Reward(value=reward_value, explanation=f"Actions failed with reasons: {fail_reasons}")
        
        return None