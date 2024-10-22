## Description of the Flow
The search algorithm operates in a loop, following these main steps to explore and evaluate possible actions:

1. **Selection:** Choose the next `Node` to expand using a `Selector` strategy. The selector evaluates the available nodes (expandable descendants) and selects the most promising one based on predefined criteria.

2. **Expansion:** Create a new child `Node` or select an existing unexecuted child.

3. **Simulation:**
   * **Action Generation and Execution:** Use the `Agent` to generate and execute an action for the `Node`:
     - Generate possible actions for the node
     - Create a system prompt and messages based on the node's history
     - Use a `CompletionModel` to generate action arguments
     - Execute the chosen action, updating the node's `FileContext` and creating an `Observation`
   * **Reward Evaluation:** If a `ValueFunction` is defined, evaluate the outcome of the action execution, assigning a reward to the `Node`.

4. **Backpropagation:** Propagate the obtained reward back up the tree, updating the value estimates and visit counts of ancestor nodes.

When the search process finishes (based on predefined stopping criteria), the algorithm determines the best solution found using a `Discriminator`, which assesses the nodes based on their rewards and other factors.

### Node
   
```python
class Node(BaseModel):
    node_id: int = Field(..., description="The unique identifier of the node")
    parent: Optional['Node'] = Field(None, description="The parent node")
    children: List['Node'] = Field(default_factory=list, description="The child nodes")
    is_duplicate: bool = Field(False, description="Flag to indicate if the node is a duplicate")
    action: Optional[ActionArguments] = Field(None, description="The action associated with the node")
    observation: Optional[Observation] = Field(None, description="The observation from the executed action")
    reward: Optional[Reward] = Field(None, description="The reward of the node")
    visits: int = Field(0, description="The number of times the node has been visited")
    value: float = Field(0.0, description="The total value (reward) of the node")
    file_context: Optional[FileContext] = Field(None, description="The file context state associated with the node")
    message: Optional[str] = Field(None, description="The message associated with the node")
    feedback: Optional[str] = Field(None, description="Feedback provided to the node")
    completions: Dict[str, Completion] = Field(default_factory=dict, description="The completions used in this node")
``` 

### Selector

```python
class Selector(ABC):
    @abstractmethod
    def select(self, nodes: List[Node]) -> Optional[Node]:
        """
        Selects the next node to expand from a list of candidate nodes.

        Args:
            nodes (List[Node]): A list of candidate nodes to select from.

        Returns:
            Optional[Node]: The selected node, or None if no suitable node is found.
        """
        pass
```

### Agent
Responsible for generating and executing actions:

```python
class Agent(BaseModel):
    def run(self, node: Node):
        self._generate_action(node)
        self._execute_action(node)
```

### Action
Defines the structure and execution logic for different types of actions:

```python
class Action(BaseModel, ABC):
    args_schema: Type[ActionArguments]

    def execute(self, args: ActionArguments, file_context: FileContext) -> Observation:
        message = self._execute(file_context=file_context)
        return Observation.create(message)
```

### ActionArguments
Defines the arguments required for each action type:

```python
class ActionArguments(OpenAISchema, ABC):
    scratch_pad: str = Field(..., description="Your reasoning for the action.")
```

### Observation
Represents the result of an action execution:

```python
class Observation(BaseModel):
    message: str
    extra: Optional[str] = None
    terminal: bool = False
    expect_correction: bool = False
    properties: Optional[Dict[str, Any]] = None
    execution_completion: Optional[Completion] = None
```

### ValueFunction

```python
class ValueFunction(ABC):
    @abstractmethod
    def get_reward(self, node: Node) -> Reward:
        """
        Evaluates the node's outcome and assigns a reward.

        Args:
            node (Node): The node to evaluate.

        Returns:
            Reward: The calculated reward
        """
        pass
``` 

#### Reward

```python
class Reward(OpenAISchema):
    explanation: Optional[str] = Field(None, description="An explanation and the reasoning behind your decision.")
    feedback: Optional[str] = Field(None, description="Feedback to the alternative branch.")
    value: int = Field(..., description="As ingle integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue")
``` 

### FeedbackGenerator

```python

class FeedbackGenerator(ABC):
    @abstractmethod
    def generate_feedback(self, node: Node) -> str:
        """
        Generates feedback for a given node to inform the next action.

        Args:
            node (Node): The node for which to generate feedback.

        Returns:
            str: The feedback message to be used in action building.
        """
        pass
``` 

### Discriminator

```python
class Discriminator(ABC):
    @abstractmethod
    def select(self, nodes: List['Node']) -> Optional['Node']:
        """
        Selects the best node from a list of nodes based on specific criteria.

        Args:
            nodes (List[Node]): A list of nodes to evaluate.

        Returns:
            Optional[Node]: The node deemed the best, or None if no suitable node is found.
        """
        pass
``` 
