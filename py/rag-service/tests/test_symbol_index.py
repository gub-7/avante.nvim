"""Tests for the symbol extraction and search index."""

from __future__ import annotations


def test_replace_symbols_for_file(fake_repo):
    """Symbols should be written to the DB and be query-able afterwards."""
    from libs.utils import path_to_uri
    from rag.symbol_index import replace_symbols_for_file, search_symbols

    file_path = fake_repo / "src" / "main.py"
    file_uri = path_to_uri(file_path)
    resource_uri = path_to_uri(fake_repo)
    content = file_path.read_text()

    count = replace_symbols_for_file(
        file_uri=file_uri,
        resource_uri=resource_uri,
        content=content,
    )
    assert count > 0, "at least one symbol must be extracted"


def test_search_symbol_foo(fake_repo):
    """The function 'foo' defined in main.py must appear in the symbol DB."""
    from libs.utils import path_to_uri
    from rag.symbol_index import replace_symbols_for_file, search_symbols

    file_path = fake_repo / "src" / "main.py"
    file_uri = path_to_uri(file_path)
    resource_uri = path_to_uri(fake_repo)
    content = file_path.read_text()

    replace_symbols_for_file(file_uri=file_uri, resource_uri=resource_uri, content=content)

    results = search_symbols(resource_uri, "foo", kinds=["function"])
    assert any(r["symbol_name"] == "foo" for r in results), (
        f"expected 'foo' in results, got: {results}"
    )

