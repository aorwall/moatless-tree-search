from abc import ABC, abstractmethod
from typing import Optional, Dict

from pydantic import BaseModel, Field


class Repository(BaseModel, ABC):
    @abstractmethod
    def get_file_content(self, file_path: str) -> Optional[str]:
        pass

    def file_exists(self, file_path: str) -> bool:
        return True

    def save_file(self, file_path: str, updated_content: str):
        pass

    def is_directory(self, file_path: str) -> bool:
        return False


class InMemRepository(Repository):
    files: Dict[str, str] = Field(default_factory=dict)

    def get_file_content(self, file_path: str) -> Optional[str]:
        return self.files.get(file_path)

    def file_exists(self, file_path: str) -> bool:
        return file_path in self.files

    def save_file(self, file_path: str, updated_content: str):
        self.files[file_path] = updated_content

    def model_dump(self) -> Dict:
        return {"files": self.files}

    @classmethod
    def model_validate(cls, obj: Dict):
        return cls(files=obj.get("files", {}))
