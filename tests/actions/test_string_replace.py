import pytest
from unittest.mock import Mock

from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository


@pytest.fixture
def repository():
    repo = InMemRepository()
    repo.save_file("test.py", """def hello():
    message = "Hello World"
    print(message)
""")
    return repo

@pytest.fixture
def file_context(repository):
    context = FileContext(repo=repository)
    # Add file to context to ensure it's available
    context.add_file("test.py", show_all_spans=True)
    return context


def test_string_replace_basic(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='message = "Hello World"',
        new_str='message = "Hello Universe"',
        scratch_pad="Updating greeting message"
    )
    
    observation = action.execute(args, file_context)

    assert observation.properties["success"]
    content = file_context.get_file("test.py").content
    assert 'message = "Hello Universe"' in content
    assert "def hello():" in content  # Verify the rest of the file is intact
    assert "print(message)" in content
    assert "diff" in observation.properties

def test_string_replace_not_found(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='not_existing_string',
        new_str='new_string',
        scratch_pad="Trying to replace non-existent string"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "string_not_found"
    assert observation.expect_correction

def test_string_replace_multiple_occurrences(repository, file_context):
    # Create file with multiple occurrences
    repository.save_file("test2.py", """def hello():
        message = "test"
        print(message)
        message = "test"
    """)
    file_context.add_file("test2.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test2.py",
        old_str='    message = "test"',  # Include proper indentation in search string
        new_str='    message = "updated"',
        scratch_pad="Updating test messages"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "multiple_occurrences"
    assert observation.expect_correction

def test_string_replace_file_not_found(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="nonexistent.py",
        old_str='old_string',
        new_str='new_string',
        scratch_pad="Trying to modify non-existent file"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "file_not_found"
    assert observation.expect_correction

def test_string_replace_same_string(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='message = "Hello World"',
        new_str='message = "Hello World"',
        scratch_pad="Trying to replace with same string"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "no_changes"
    assert observation.expect_correction

def test_string_replace_with_indentation(repository, file_context):
    # Create file with indented content - note the proper indentation
    repository.save_file("test3.py", """class Test:
    def method(self):
        value = "old"
        return value
""")
    file_context.add_file("test3.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test3.py",
        old_str='        value = "old"',  # Include proper indentation in search string
        new_str='        value = "new"',
        scratch_pad="Updating indented value"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["success"]
    content = file_context.get_file("test3.py").content
    assert '        value = "new"' in content
    assert "class Test:" in content  # Verify the rest of the file is intact
    assert "def method(self):" in content
    assert "diff" in observation.properties

def test_string_replace_with_newlines(repository, file_context):
    # Create file with multiline content - note the proper indentation
    repository.save_file("test4.py", """def old_function():
    print("line1")
    print("line2")
""")
    file_context.add_file("test4.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test4.py",
        old_str='''def old_function():
    print("line1")
    print("line2")''',
        new_str='''def new_function():
    print("new_line1")
    print("new_line2")''',
        scratch_pad="Replacing entire function"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["success"]
    content = file_context.get_file("test4.py").content
    assert 'def new_function():' in content
    assert '    print("new_line1")' in content  # Check indented content
    assert '    print("new_line2")' in content
    assert "diff" in observation.properties