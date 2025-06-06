import logging
from typing import Any, Dict, Optional

from pydantic import Field

from moatless.node import Node, Reward
from moatless.completion.base import BaseCompletionModel
from moatless.value_function.base import BaseValueFunction

from swesearch.value_function.fail_reasons import FailReasonsValueFunction
from swesearch.value_function.finish_evaluator import FinishEvaluator
from swesearch.value_function.repeated_action import RepeatedActionValueFunction
from swesearch.value_function.tests import TestsValueFunction
from swesearch.value_function.think_evaluator import ThinkEvaluator
from swesearch.value_function.action_evaluator import ActionEvaluator

logger = logging.getLogger(__name__)

class SweSearchValueFunction(BaseValueFunction):
    completion_model: BaseCompletionModel = Field(
        description="Completion model to be used for generating completions"
    )
    
    use_action_evaluator: bool = Field(
        default=False,
        description="If True, uses the action evaluator to get the reward"
    )
    
    async def get_reward(self, node: Node) -> Optional[Reward]:
        if self.has_action_step(node, "Finish"):
            finish_func = FinishEvaluator(completion_model=self.completion_model.clone())
            finish_reward = await finish_func.get_reward(node)
            if finish_reward:
                logger.info(f"Finish reward: {finish_reward}")
                return finish_reward
            
        if self.has_action_step(node, "Think"):
            think_func = ThinkEvaluator(completion_model=self.completion_model.clone())
            think_reward = await think_func.get_reward(node)
            if think_reward:
                logger.info(f"Think reward: {think_reward}")
                return think_reward
            
        if self.use_action_evaluator:
            action_evaluator_func = ActionEvaluator(completion_model=self.completion_model.clone())
            action_evaluator_reward = await action_evaluator_func.get_reward(node)
            if action_evaluator_reward:
                logger.info(f"Action evaluator reward: {action_evaluator_reward}")
                return action_evaluator_reward
        
        fail_reason_func = FailReasonsValueFunction()
        fail_reason_reward = await fail_reason_func.get_reward(node)
        if fail_reason_reward:
            logger.info(f"Fail reason reward: {fail_reason_reward}")
            return fail_reason_reward
        
        if self.has_action_step(node, "RunTests"):
            tests_func = TestsValueFunction()
            tests_reward = await tests_func.get_reward(node)
            if tests_reward:
                logger.info(f"Tests reward: {tests_reward}")
                return tests_reward
        
        repeated_action_func = RepeatedActionValueFunction()
        repeated_action_reward = await repeated_action_func.get_reward(node)
        if repeated_action_reward:
            logger.info(f"Repeated action reward: {repeated_action_reward}")
            return repeated_action_reward
        
        logger.info(f"No reward found for node {node.node_id}")
        return None

    def has_action_step(self, node: Node, action_name: str) -> bool:
        for action_step in node.action_steps:
            if action_step.action.name == action_name:
                return True
        return False
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        if self.completion_model:
            dump["completion_model"] = self.completion_model.model_dump(**kwargs)
        return dump
    
    @classmethod
    def model_validate(cls, obj: Any) -> "SweSearchValueFunction":
        if isinstance(obj, dict):
            obj = obj.copy()
            if "completion_model" in obj:
                obj["completion_model"] = BaseCompletionModel.from_dict(obj.pop("completion_model"))

            return super().model_validate(obj)

        return obj