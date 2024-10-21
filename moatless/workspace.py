import logging
from datetime import datetime
from typing import Any, Optional, Dict

from moatless.codeblocks.parser.python import PythonParser
from moatless.file_context import FileContext
from moatless.index import IndexSettings
from moatless.index.code_index import CodeIndex
from moatless.repository import FileRepository, GitRepository
from moatless.schema import FileWithSpans, TestResult
from moatless.verify.verify import RuntimeEnvironment

_parser = PythonParser()

logger = logging.getLogger(__name__)


class Workspace:
    def __init__(
        self,
        file_repo: FileRepository,
        index_dir: Optional[str] = None,
        index_settings: IndexSettings | None = None,
        max_results: int = 25,
        code_index: CodeIndex | None = None,
        # verification_job: Optional[str] = "pylint",
        max_file_context_tokens: int = 4000,
        file_context: FileContext | None = None,
        runtime_environment: RuntimeEnvironment | None = None,
    ):
        self.file_repo = file_repo

        if code_index:
            self.code_index = code_index
        elif index_dir:
            try:
                self.code_index = CodeIndex.from_persist_dir(
                    index_dir, file_repo=file_repo, max_results=max_results
                )
            except FileNotFoundError:
                logger.info("No index found. Creating a new index.")
                code_index = CodeIndex(
                    file_repo=file_repo,
                    settings=index_settings,
                    max_results=max_results,
                )
                code_index.run_ingestion()
                code_index.persist(index_dir)
                self.code_index = code_index
        else:
            self.code_index = None

        if runtime_environment:
            self.runtime = runtime_environment
        # elif verification_job == "maven" and self.file_repo:
        #     self.verifier = MavenVerifier(self.file_repo.path)
        # elif verification_job == "pylint" and self.file_repo:
        #     self.verifier = PylintVerifier(self.file_repo.path)
        else:
            self.runtime = None

        self._file_context = None

    @classmethod
    def from_dirs(
        cls,
        git_repo_url: Optional[str] = None,
        commit: Optional[str] = None,
        repo_path: Optional[str] = None,
        max_file_context_tokens: int = 4000,
        **kwargs,
    ):
        if git_repo_url:
            file_repo = GitRepository.from_repo(
                git_repo_url=git_repo_url, repo_path=repo_path, commit=commit
            )
        elif repo_path:
            file_repo = FileRepository(repo_path)
        else:
            raise ValueError("Either git_repo_url or repo_dir must be provided.")

        return cls(
            file_repo=file_repo,
            max_file_context_tokens=max_file_context_tokens,
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict,
        base_repo_dir: str | None = None,
        repo_path: str | None = None,
        **kwargs,
    ):
        if "repository" not in data:
            raise ValueError("Missing repository key")

        if base_repo_dir and not repo_path:
            # last dir in data["repository"].get("repo_path")
            repo_dir_name = data["repository"].get("repo_path").split("/")[-1]
            repo_path = f"{base_repo_dir}/{repo_dir_name}"
        elif not repo_path:
            repo_path = data["repository"].get("repo_path")

        if data["repository"].get("git_repo_url"):
            file_repo = GitRepository.from_repo(
                git_repo_url=data["repository"].get("git_repo_url"),
                repo_path=repo_path,
                commit=data["repository"].get("commit"),
            )
        elif data["repository"].get("path"):
            file_repo = FileRepository(repo_path)
        else:
            raise ValueError("Either git_repo_url or path must be provided.")

        file_context = FileContext(
            repo=file_repo, max_tokens=data["file_context"].get("max_tokens")
        )
        file_context.load_files_from_dict(data["file_context"].get("files", []))

        if data.get("code_index", {}).get("index_name"):
            code_index = CodeIndex.from_index_name(
                data["code_index"].get("index_name"), file_repo=file_repo
            )
        else:
            code_index = None

        return cls(
            file_repo=file_repo,
            file_context=file_context,
            code_index=code_index,
            **kwargs,
        )

    def restore_from_snapshot(self, snapshot: dict):
        self.file_repo.restore_from_snapshot(snapshot["repository"])
        self._file_context.restore_from_snapshot(snapshot["file_context"])

    def dict(self):
        if not self.file_repo:
            return {}
        return {
            "repository": self.file_repo.dict(),
            "file_context": self.file_context.model_dump(
                exclude_none=True, exclude_unset=True
            ),
            "code_index": self.code_index.dict() if self.code_index else None,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "repository": self.file_repo.snapshot(),
            "file_context": self.file_context.snapshot(),
        }

    def create_file_context(
        self,
        files_with_spans: list[FileWithSpans] | None = None,
        max_tokens: int = 4000,
    ):
        file_context = FileContext(self.file_repo, max_tokens=max_tokens)
        if files_with_spans:
            file_context.add_files_with_spans(files_with_spans)
        return file_context

    @property
    def file_context(self):
        return self._file_context

    def get_file(self, file_path):
        return self.file_repo.get_file(file_path)

    def run_tests(
        self, file_context: FileContext, test_files: list[str]
    ) -> list[TestResult]:
        if self.runtime:
            return self.runtime.run_tests(file_context, test_files)

        return []

    def evaluate(self):
        if self.runtime:
            return self.runtime.evaluate()
