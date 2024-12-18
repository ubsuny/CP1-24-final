import final
import pytest

def test_fahrenheit_to_kelvin():
    assert final.fahrenheit_to_kelvin(32) == pytest.approx(273.15)
    assert final.fahrenheit_to_kelvin(212) == pytest.approx(373.15)

def test_parse_temperature(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("Environment temperature: 72F")
    assert final.parse_temperature(md_file) == 72

def test_list_markdown_files(tmp_path):
    (tmp_path / "test1.md").write_text("File 1")
    (tmp_path / "ignore.txt").write_text("File 2")
    result = final.list_markdown_files(tmp_path, "test")
    assert "test1.md" in result
