from pydantic import BaseModel
from enum import Enum
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


class InstanceItemDTO(BaseModel):
    instanceId: str
    status: Literal["pending", "running", "completed", "failed", "resolved", "error"]
    duration: Optional[float] = None
    resolved: Optional[bool] = None
    error: Optional[str] = None
    iterations: Optional[int] = None
    completionCost: Optional[float] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    resolutionRate: Optional[float] = None
    flags: List[str] = []
    splits: List[str] = []

class EvaluationSettingsDTO(BaseModel):
    model: str
    temperature: float
    maxIterations: int
    responseFormat: Literal["tool_call", "json", "react"]
    maxCost: float

class EvaluationResponseDTO(BaseModel):
    name: str | None = None
    status: Literal["pending", "running", "completed", "error"]
    isActive: bool
    settings: EvaluationSettingsDTO
    startedAt: Optional[datetime]
    totalCost: float
    promptTokens: int
    completionTokens: int
    totalInstances: int
    completedInstances: int
    errorInstances: int
    resolvedInstances: int
    failedInstances: int
    instances: List[InstanceItemDTO]


class EvaluationListItemDTO(BaseModel):
    """Represents an evaluation item in the list view."""
    name: str
    status: Literal["pending", "running", "completed", "error"]
    model: str
    maxExpansions: int
    startedAt: Optional[datetime]
    totalInstances: int
    completedInstances: int
    errorInstances: int
    resolvedInstances: int
    isActive: bool
    date: Optional[datetime] = None  # For easier date-based sorting/filtering
    resolutionRate: float = 0.0  # resolved/total instances
    totalCost: float = 0.0
    promptTokens: int = 0
    completionTokens: int = 0
    resolvedByDollar: float = 0.0  # resolved instances per dollar spent

class EvaluationListResponseDTO(BaseModel):
    """Response model for list evaluations endpoint."""
    evaluations: List[EvaluationListItemDTO]

class UsageDTO(BaseModel):
    """Usage information for a completion."""
    completionCost: Optional[float] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    cachedTokens: Optional[int] = None

class CompletionDTO(BaseModel):
    """Completion information."""
    type: str
    usage: Optional[UsageDTO] = None
    tokens: str

class ObservationDTO(BaseModel):
    """Observation information."""
    message: Optional[str] = None
    summary: Optional[str] = None
    properties: Dict[str, Any] = {}
    expectCorrection: bool = False

class ActionDTO(BaseModel):
    """Action information."""
    name: str
    thoughts: Optional[str] = None
    properties: Dict[str, Any] = {}

class ActionStepDTO(BaseModel):
    """Represents a single action step."""
    thoughts: Optional[str] = None
    action: ActionDTO
    observation: Optional[ObservationDTO] = None
    completion: Optional[CompletionDTO] = None

class FileContextSpanDTO(BaseModel):
    """Represents a span in a file context."""
    span_id: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    tokens: Optional[int] = None
    pinned: bool = False

class FileContextFileDTO(BaseModel):
    """Represents a file in the file context."""
    file_path: str
    patch: Optional[str] = None
    spans: List[FileContextSpanDTO] = []
    show_all_spans: bool = False
    tokens: Optional[int] = None
    is_new: bool = False
    was_edited: bool = False

class FileContextDTO(BaseModel):
    """File context information."""
    summary: str
    testSummary: Optional[str] = None
    testResults: Optional[List[Dict[str, Any]]] = None
    patch: Optional[str] = None
    files: List[FileContextFileDTO] = []

class NodeDTO(BaseModel):
    """Node information in the tree."""
    nodeId: int
    actionSteps: List[ActionStepDTO] = []
    assistantMessage: Optional[str] = None
    userMessage: Optional[str] = None
    completionUsage: Optional[UsageDTO] = None
    completions: Dict[str, CompletionDTO] = {}
    fileContext: Optional[FileContextDTO] = None
    warnings: List[str] = []
    errors: List[str] = []

class InstanceResponseDTO(BaseModel):
    """Response model for tree visualization endpoint."""
    nodes: List[NodeDTO]
    totalNodes: int
    instance: Optional[Dict[str, Any]] = None
    evalResult: Optional[Dict[str, Any]] = None
    status: str
    duration: Optional[float] = None
    resolved: Optional[bool] = None
    error: Optional[str] = None
    iterations: Optional[int] = None
    completionCost: Optional[float] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    resolutionRate: Optional[float] = None
    splits: List[str] = []
    flags: List[str] = []
