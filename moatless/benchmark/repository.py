import os
import json
import shutil
import logging
import glob
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pathlib import Path

from moatless.benchmark.schema import Evaluation, EvaluationInstance, DateTimeEncoder, InstanceStatus

logger = logging.getLogger(__name__)

class EvaluationRepository(ABC):
    """Abstract base class for evaluation persistence."""
    
    @abstractmethod
    def save_evaluation(self, evaluation: Evaluation) -> None:
        """Save an evaluation."""
        pass
            
    @abstractmethod
    def load_evaluation(self, evaluation_name: str) -> Evaluation:
        """Load an evaluation."""
        pass
            
    @abstractmethod
    def save_instance(self, evaluation_name: str, instance: EvaluationInstance) -> None:
        """Save an instance."""
        pass
            
    @abstractmethod
    def load_instance(self, evaluation_name: str, instance_id: str) -> Optional[EvaluationInstance]:
        """Load an instance."""
        pass
            
    @abstractmethod
    def delete_instance(self, evaluation_name: str, instance_id: str) -> None:
        """Delete an instance."""
        pass

    @abstractmethod
    def list_instances(self, evaluation_name: str) -> List[EvaluationInstance]:
        """List all instances for an evaluation."""
        pass

    @abstractmethod
    def list_evaluations(self) -> List[str]:
        """List all evaluation names from storage."""
        pass


class EvaluationFileRepository(EvaluationRepository):
    """File-based implementation of evaluation repository."""
    
    def __init__(self, evaluations_dir: str):
        self.evaluations_dir = evaluations_dir
        
    def get_evaluation_dir(self, evaluation_name: str) -> str:
        """Get the directory path for an evaluation."""
        return os.path.join(self.evaluations_dir, evaluation_name)
        
    def get_instance_dir(self, evaluation_name: str, instance_id: str) -> str:
        """Get the directory path for an instance within an evaluation."""
        return os.path.join(self.get_evaluation_dir(evaluation_name), instance_id)
        
    def save_evaluation(self, evaluation: Evaluation) -> None:
        """Save an evaluation to disk."""
        eval_dir = self.get_evaluation_dir(evaluation.evaluation_name)
        if not os.path.exists(eval_dir):
            logger.debug(f"Creating evaluation directory: {eval_dir}")
            os.makedirs(eval_dir, exist_ok=True)
            
        eval_file = os.path.join(eval_dir, "evaluation.json")
        logger.info(f"Saving evaluation {evaluation.evaluation_name} to {eval_file}")
        with open(eval_file, "w") as f:
            json.dump(evaluation.model_dump(), f, cls=DateTimeEncoder, indent=2)
            
    def load_evaluation(self, evaluation_name: str) -> Evaluation | None:
        """Load an evaluation from disk."""
        eval_path = os.path.join(self.get_evaluation_dir(evaluation_name), "evaluation.json")
        logger.debug(f"Attempting to load evaluation from: {eval_path}")
        if not os.path.exists(eval_path):
            logger.warning(f"Evaluation file not found: {eval_path}")
            return None
            
        try:
            logger.debug(f"Reading evaluation file: {eval_path}")
            with open(eval_path, 'r') as f:
                data = json.load(f)
                evaluation = Evaluation.model_validate(data)
                logger.debug(f"Successfully loaded evaluation {evaluation_name} with status {evaluation.status}")
                return evaluation
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in evaluation file {eval_path}: {e}")
            raise ValueError(f"Invalid JSON in evaluation file: {e}")
        except Exception as e:
            logger.error(f"Error loading evaluation from {eval_path}: {e}")
            raise ValueError(f"Error loading evaluation: {e}")
            
    def save_instance(self, evaluation_name: str, instance: EvaluationInstance) -> None:
        """Save an instance to disk."""
        instance_dir = self.get_instance_dir(evaluation_name, instance.instance_id)
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir, exist_ok=True)
            
        instance_file = os.path.join(instance_dir, "instance.json")
        with open(instance_file, "w") as f:
            json.dump(instance.model_dump(), f, cls=DateTimeEncoder, indent=2)
            
    def load_instance(self, evaluation_name: str, instance_id: str) -> Optional[EvaluationInstance]:
        """Load an instance from disk."""
        instance_file = os.path.join(self.get_instance_dir(evaluation_name, instance_id), "instance.json")
        if not os.path.exists(instance_file):
            return None
            
        try:
            with open(instance_file) as f:
                data = json.load(f)

            instance = EvaluationInstance.model_validate(data)
            
            # Check instance_response.json for status
            instance_response_file = os.path.join(self.get_instance_dir(evaluation_name, instance_id), "instance_response.json")
            if os.path.exists(instance_response_file):
                try:
                    with open(instance_response_file) as f:
                        response_data = json.load(f)
                        from moatless_tools.schema import InstanceResponseDTO
                        instance_response = InstanceResponseDTO.model_validate(response_data)
                        if instance_response.status:
                            try:
                                # Map common status values to InstanceStatus enum
                                status_map = {
                                    "completed": InstanceStatus.COMPLETED,
                                    "error": InstanceStatus.ERROR,
                                    "failed": InstanceStatus.ERROR,
                                    "pending": InstanceStatus.PENDING
                                }
                                new_status = status_map.get(instance_response.status.lower())
                                if new_status != instance.status:
                                    instance.status = new_status
                                    logger.info(f"Updated instance {instance_id} status to {new_status} from response")
                                else:
                                    logger.warning(f"Unknown status value in instance_response: {instance_response.status}")
                            except Exception as e:
                                logger.warning(f"Failed to map instance response status: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load instance response for status update: {e}")
            return instance
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error loading instance state from {instance_file}: {e}")
            return None
            
    def delete_instance(self, evaluation_name: str, instance_id: str) -> None:
        """Delete an instance directory."""
        instance_dir = self.get_instance_dir(evaluation_name, instance_id)
        if os.path.exists(instance_dir):
            shutil.rmtree(instance_dir)

    def list_instances(self, evaluation_name: str) -> List[EvaluationInstance]:
        """List all instances for an evaluation."""
        eval_dir = self.get_evaluation_dir(evaluation_name)
        logger.info(f"Listing instances in directory: {eval_dir}")
        if not os.path.exists(eval_dir):
            logger.warning(f"Evaluation directory does not exist: {eval_dir}")
            return []

        instance_dirs = glob.glob(os.path.join(eval_dir, "*/"))
        logger.info(f"Found {len(instance_dirs)} potential instance directories to scan")
        
        instances = []
        for instance_dir in instance_dirs:
            instance_id = os.path.basename(os.path.dirname(instance_dir))
            logger.debug(f"Found instance directory: {instance_dir} with ID: {instance_id}")
            instance = self.load_instance(evaluation_name, instance_id)
            if instance:
                logger.debug(f"Successfully loaded instance {instance_id} with status {instance.status}")
                instances.append(instance)
            else:
                logger.warning(f"Failed to load instance from directory: {instance_dir}")

        logger.info(f"Found {len(instances)} instances for evaluation {evaluation_name}")
        return instances

    def list_evaluations(self) -> List[str]:
        """List all evaluation names from disk."""
        if not os.path.exists(self.evaluations_dir):
            logger.debug(f"Evaluations directory does not exist: {self.evaluations_dir}")
            return []
            
        # Use os.listdir to get all directories and filter for those that have evaluation.json
        all_dirs = os.listdir(self.evaluations_dir)
        logger.debug(f"All directories in {self.evaluations_dir}: {all_dirs}")
        
        eval_dirs = [
            d for d in all_dirs
            if os.path.isdir(os.path.join(self.evaluations_dir, d)) and
            os.path.exists(os.path.join(self.evaluations_dir, d, "evaluation.json"))
        ]
        logger.debug(f"Found evaluation directories: {eval_dirs}")
        return sorted(eval_dirs)  # Sort for consistent ordering 