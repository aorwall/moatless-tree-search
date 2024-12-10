import pytest

from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository
from moatless.selector.similarity import FileContextSimilarity


@pytest.fixture
def mock_repo():
    return InMemRepository()

def test_similarity_identical_contexts(mock_repo):
    # Create two identical FileContext instances
    context_a = FileContext(repo=mock_repo)
    context_b = FileContext(repo=mock_repo)

    # Add the same files and spans to both contexts
    context_a.add_file('file1.py')
    context_a.add_span_to_context('file1.py', 'span1')
    context_b.add_file('file1.py')
    context_b.add_span_to_context('file1.py', 'span1')

    similarity = FileContextSimilarity().calculate_similarity(context_a, context_b)
    assert similarity == 1.0, "Identical contexts should have a similarity of 1.0"

def test_similarity_different_contexts(mock_repo):
    # Create two different FileContext instances
    context_a = FileContext(repo=mock_repo)
    context_b = FileContext(repo=mock_repo)

    # Add different files and spans to each context
    context_a.add_file('file1.py')
    context_a.add_span_to_context('file1.py', 'span1')
    context_b.add_file('file2.py')
    context_b.add_span_to_context('file2.py', 'span2')

    similarity = FileContextSimilarity().calculate_similarity(context_a, context_b)
    assert similarity == 0.0, "Completely different contexts should have a similarity of 0.0"

def test_partial_similarity(mock_repo):
    # Create partially similar FileContext instances
    context_a = FileContext(repo=mock_repo)
    context_b = FileContext(repo=mock_repo)

    # Add overlapping and non-overlapping spans
    context_a.add_file('file1.py')
    context_a.add_span_to_context('file1.py', 'span1')
    context_a.add_span_to_context('file1.py', 'span2')
    context_b.add_file('file1.py')
    context_b.add_span_to_context('file1.py', 'span1')
    context_b.add_span_to_context('file1.py', 'span3')

    similarity = FileContextSimilarity().calculate_similarity(context_a, context_b)
    assert 0.0 < similarity < 1.0, "Partially similar contexts should have a similarity between 0.0 and 1.0"
