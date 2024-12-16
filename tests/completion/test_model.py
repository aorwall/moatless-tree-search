import pytest
from pydantic import ValidationError

from moatless.actions.create_file import CreateFileArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.completion.model import StructuredOutput


def test_string_replace_xml_validation():
    # Test valid XML format
    valid_xml = """
<path>test/file.py</path>
<old_str>def old_function():
    pass</old_str>
<new_str>def new_function():
    return True</new_str>
"""
    
    args = StringReplaceArgs.model_validate_xml(valid_xml)
    assert args.path == "test/file.py"
    assert args.old_str == "def old_function():\n    pass"
    assert args.new_str == "def new_function():\n    return True"

    # Test invalid XML - missing closing tag
    invalid_xml = """
<path>test/file.py</path>
<old_str>some code</old_str>
<new_str>new code
"""
    with pytest.raises(ValidationError):
        StringReplaceArgs.model_validate_xml(invalid_xml)

    # Test invalid XML - wrong tag names
    wrong_tags_xml = """
<path>test/file.py</path>
<old_str>some code</old_str>
<file_text>new code</file_text>
"""
    with pytest.raises(ValidationError):
        StringReplaceArgs.model_validate_xml(wrong_tags_xml)

def test_create_file_xml_validation():
    # Test valid XML format
    valid_xml = """
<path>new/test/file.py</path>
<file_text>def test_function():
    return True</file_text>
"""
    
    args = CreateFileArgs.model_validate_xml(valid_xml)
    assert args.path == "new/test/file.py"
    assert args.file_text == "def test_function():\n    return True"

def test_string_replace_indentation():
    # Test indentation preservation in XML format
    xml_with_indentation = """
<path>test/file.py</path>
<old_str>        data = StringIO(data)
        for obj in serializers.deserialize("json", data, using=self.connection.alias):
            obj.save()</old_str>
<new_str>        data = StringIO(data)
        with transaction.atomic(using=self.connection.alias):
            for obj in serializers.deserialize("json", data, using=self.connection.alias):
                obj.save()</new_str>
"""
    
    args = StringReplaceArgs.model_validate_xml(xml_with_indentation)
    assert args.path == "test/file.py"
    assert args.old_str == "        data = StringIO(data)\n        for obj in serializers.deserialize(\"json\", data, using=self.connection.alias):\n            obj.save()"
    assert args.new_str == "        data = StringIO(data)\n        with transaction.atomic(using=self.connection.alias):\n            for obj in serializers.deserialize(\"json\", data, using=self.connection.alias):\n                obj.save()"

def test_structured_output_name_and_description():
    # Test class with Config.title and docstring
    class TestWithConfig(StructuredOutput):
        """This is a test description."""
        class Config:
            title = "CustomTitle"

    assert TestWithConfig.name == "CustomTitle"
    assert TestWithConfig.description == "This is a test description."

    # Test class without Config.title but with docstring
    class TestWithoutConfig(StructuredOutput):
        """Another test description."""
        pass

    assert TestWithoutConfig.name == "TestWithoutConfig"
    assert TestWithoutConfig.description == "Another test description."

    # Test class without Config.title or docstring
    class TestBare(StructuredOutput):
        pass

    assert TestBare.name == "TestBare"
    assert TestBare.description == ""
