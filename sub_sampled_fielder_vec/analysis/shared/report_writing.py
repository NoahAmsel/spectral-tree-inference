"""Generate markdown reports."""

from pathlib import Path


def save_markdown_report(content: str, output_path: Path):
    """
    Save a markdown report to file.

    Args:
        content: Markdown content string
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Saved report: {output_path}")
