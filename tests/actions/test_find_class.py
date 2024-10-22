from moatless.actions.find_class import FindClass, FindClassArgs
from moatless.benchmark.swebench import create_repository, create_index
from moatless.benchmark.utils import get_moatless_instance
from moatless.file_context import FileContext


def test_find_class__too_large():
    instance_id = "django__django-13658"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)

    action = FindClass(
        repository=repository, code_index=code_index
    )

    args = FindClassArgs(
        scratch_pad="",
        class_name="ManagementUtility",
    )

    output = action.execute(args, file_context)
    print(output.extra)
    assert len(file_context.files) == 0
    assert output.extra.startswith("The class is too large.")
    assert "ManagementUtility" in output.extra
    assert output.message == "Found 1 classes."


def test_find_class__wrong_file_pattern():
    instance_id = "django__django-13658"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)

    action = FindClass(
        repository=repository, code_index=code_index
    )

    action_args = FindClassArgs(
        scratch_pad="", class_name="ManagementUtility", file_pattern="**/foo/**"
    )

    output = action.execute(action_args, file_context)
    print(output.extra)
    assert len(file_context.files) == 0
    assert (
        output.message
        == "No files found for file pattern **/foo/**. But found 1 alternative suggestions."
    )
    assert "ManagementUtility" in output.extra
