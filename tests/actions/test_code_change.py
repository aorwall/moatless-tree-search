from moatless.actions.code_change import RequestCodeChange
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.completion import CompletionModel
from moatless.file_context import FileContext
import pytest
from moatless.actions.code_change import RequestCodeChangeArgs, ChangeType
from moatless.repository.repository import Repository, InMemRepository


class MockCompletionModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    def create_text_completion(self, messages, system_prompt):
        return self.mock_response, None


def test_one_line_change():
    instance = get_moatless_instance("django__django-13658")
    workspace = create_workspace(instance)

    workspace.file_context.add_spans_to_context(
        "django/core/management/__init__.py",
        span_ids=[
            "ManagementUtility.execute",
            "ManagementUtility.__init__",
            "ManagementUtility.fetch_command",
            "ManagementUtility.autocomplete",
            "ManagementUtility.main_help_text",
        ],
    )

    request_code_change = RequestCodeChange(
        scratch_pad="Change the instantiation of CommandParser to use self.prog_name for the prog argument.",
        file_path="django/core/management/__init__.py",
        instructions="Change the instantiation of CommandParser to use self.prog_name for the prog argument.",
        pseudo_code="parser = CommandParser(prog=self.prog_name, usage='%(prog)s subcommand [options] [args]', add_help=False, allow_abbrev=False)",
        change_type="modification",
        start_line=347,
        end_line=347,
        _workspace=workspace,
        _completion_model=CompletionModel(model="gpt-4o-mini-2024-07-18"),
    )
    request_code_change._workspace = workspace

    result = request_code_change.execute(workspace.file_context)
    print(result.message)


def test_empty_search():
    instance = get_moatless_instance("sympy__sympy-23262")
    workspace = create_workspace(instance)

    workspace.file_context.add_spans_to_context(
        "sympy/utilities/lambdify.py", span_ids=["lambdify"]
    )

    request_code_change = RequestCodeChange(
        scratch_pad="Change the instantiation of CommandParser to use self.prog_name for the prog argument.",
        file_path="django/core/management/__init__.py",
        instructions="Change the instantiation of CommandParser to use self.prog_name for the prog argument.",
        pseudo_code="parser = CommandParser(prog=self.prog_name, usage='%(prog)s subcommand [options] [args]', add_help=False, allow_abbrev=False)",
        change_type="modification",
        start_line=347,
        end_line=347,
        _workspace=workspace,
        _completion_model=CompletionModel(model="gpt-4o-mini-2024-07-18"),
    )
    request_code_change._workspace = workspace

    result = request_code_change.execute(workspace.file_context)
    print(result.message)


def test_wrong_indentation():
    instance = get_moatless_instance("sympy__sympy-15678")
    repository = create_repository(instance)

    file_context = FileContext(repo=repository)
    file_context.add_spans_to_context(
        "sympy/geometry/util.py", span_ids=["idiff"]
    )

    replace_block = """<replace>
    def idiff(eq, y, x, n=1):
        if isinstance(eq, Eq):
            eq = eq.lhs
        if is_sequence(y):
            dep = set(y)
            y = y[0]
        elif isinstance(y, Symbol):
            dep = {y}
        else:
            raise ValueError("expecting x-dependent symbol(s) but got: %s" % y)

        f = dict([(s, Function(
            s.name)(x)) for s in eq.free_symbols if s != x and s in dep])
        dydx = Function(y.name)(x).diff(x)
        eq = eq.subs(f)
        derivs = {}
        for i in range(n):
            yp = solve(eq.diff(x), dydx)[0].subs(derivs)
            if i == n - 1:
                return yp.subs([(v, k) for k, v in f.items()])
            derivs[dydx] = yp
            eq = dydx - yp
            dydx = dydx.diff(x)
</replace>
"""

    request_code_change = RequestCodeChange(
        scratch_pad="",
        file_path="sympy/geometry/util.py",
        instructions="Update the idiff function to handle cases where the equation is an instance of Eq. Extract the left-hand side of the equation for differentiation.",
        pseudo_code="if isinstance(eq, Eq):\neq = eq.lhs\n# Continue with the existing logic for idiff",
        change_type="modification",
        start_line=568,
        end_line=568,
    )

    request_code_change._completion_model = MockCompletionModel(
        mock_response=replace_block
    )

    result = request_code_change.execute(workspace.file_context)
    print(result.message)


def test_change():
    instance = get_moatless_instance("matplotlib__matplotlib-24970")
    repository = create_repository(instance)

    file_context = FileContext(repo=repository)
    file_context.add_spans_to_context(
        "lib/matplotlib/cm.py", span_ids=["ColormapRegistry.get_cmap"]
    )

    action_dict = {
        "scratch_pad": "Updating the get_cmap method to handle out-of-bound values for uint8 to avoid deprecation warnings in NumPy 1.24.",
        "file_path": "lib/matplotlib/cm.py",
        "instructions": "Modify the get_cmap method to ensure that any values being converted to uint8 are within the valid range (0-255). If they are out of bounds, handle them appropriately to avoid deprecation warnings.",
        "pseudo_code": "def get_cmap(self, cmap):\n    # ... existing code ...\n    if isinstance(cmap, str):\n        _api.check_in_list(sorted(_colormaps), cmap=cmap)\n        # Ensure cmap is within bounds for uint8\n        cmap_value = self[cmap]\n        if cmap_value < 0:\n            cmap_value = 0\n        elif cmap_value > 255:\n            cmap_value = 255\n        return cmap_value\n    # ... existing code ...",
        "change_type": "modification",
        "start_line": 183,
        "end_line": 213,
        "action_name": "RequestCodeChange",
    }

    request_code_change = RequestCodeChange(**action_dict)

    result = request_code_change.execute(file_context)
    print(result.message)


def test_request_code_change_dump_and_load():
    completion_model = CompletionModel(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )

    mock_repository = InMemRepository()

    request_code_change = RequestCodeChange(
        repository=mock_repository,
        completion_model=completion_model,
        max_tokens_in_edit_prompt=600,
        show_file_context=True
    )

    dumped_data = request_code_change.model_dump()
    dumped_data["repository"] = mock_repository

    loaded_request_code_change = RequestCodeChange.model_validate(obj=dumped_data)

    assert loaded_request_code_change._completion_model.model == "gpt-4"
    assert loaded_request_code_change._completion_model.temperature == 0.7
    assert loaded_request_code_change._completion_model.max_tokens == 1000

    assert loaded_request_code_change._repository == mock_repository
