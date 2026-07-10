from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper1.scripts.collect_tex_figures import (
    collect_tex,
    find_figure,
    output_relative_path,
)


def test_collect_tex_expands_inputs_and_nested_graphicspath(tmp_path: Path) -> None:
    child = tmp_path / "section.tex"
    child.write_text(r"\includegraphics{plot.png}", encoding="utf-8")
    main = tmp_path / "main.tex"
    main.write_text(
        "\n".join(
            [
                r"\graphicspath{{custom/}{../shared/}}",
                r"\input{section}",
            ]
        ),
        encoding="utf-8",
    )
    figure_dir = tmp_path / "custom"
    figure_dir.mkdir()
    figure = figure_dir / "plot.png"
    figure.write_bytes(b"png")

    text, graphic_paths = collect_tex(main, tmp_path, set())

    assert r"\includegraphics{plot.png}" in text
    assert graphic_paths == ["custom/", "../shared/"]
    assert find_figure("plot.png", tmp_path, graphic_paths) == figure.resolve()


def test_collect_tex_rejects_missing_input(tmp_path: Path) -> None:
    main = tmp_path / "main.tex"
    main.write_text(r"\input{missing_section}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing TeX input"):
        collect_tex(main, tmp_path, set())


def test_output_path_rejects_directory_target(tmp_path: Path) -> None:
    source = tmp_path / "plot.png"
    source.write_bytes(b"png")

    assert output_relative_path("plot", source) == Path("plot.png")
    with pytest.raises(ValueError, match="basename-only"):
        output_relative_path("subdir/plot.png", source)
