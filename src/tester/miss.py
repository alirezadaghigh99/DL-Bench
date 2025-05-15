from pathlib import Path
import json

def read_jsonl_strip(
    file_path: str | Path,
    key: str,
    part_to_remove: str | None = None,
    max_lines: int | None = None,
) -> list[str]:
    """
    Extract and optionally trim values for a given key from a JSON‑Lines file.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the *.jsonl* file (one valid JSON object per line).
    key : str
        Dictionary key whose value you want to collect.
    part_to_remove : str | None, default None
        Sub‑string to delete from each extracted value.
        ‑ If None, the value is left unchanged.
        ‑ Removal is case‑sensitive and affects **all** occurrences.
    max_lines : int | None, default None
        Optional safeguard to stop reading after *max_lines* lines.
        Useful for very large files or quick tests.

    Returns
    -------
    list[str]
        A list with one processed value per line where *key* was present.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    json.JSONDecodeError
        If a line cannot be parsed as valid JSON.
    KeyError
        If *key* is missing in a JSON object.

    Example
    -------
    >>> # input.jsonl (3 lines):
    ... # {"url": "https://example.com/page1"}
    ... # {"url": "https://example.com/page2"}
    ... # {"url": "https://example.com/page3"}
    >>> read_jsonl_strip("input.jsonl", "url", "https://")
    ['example.com/page1', 'example.com/page2', 'example.com/page3']
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    results: list[str] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if max_lines and i > max_lines:
                break
            data = json.loads(line)
            try:
                value = data[key].replace("processed_","").replace("json", "txt")
            except KeyError as exc:
                raise KeyError(f"Line {i}: key '{key}' not found") from exc

            
            results.append(value)

    return results
data = "/home/aliredaq/Desktop/DeepAgent/ICSE/data/DLEval-20240920T201632Z-001/DLEval/"
import os
datas = set(os.listdir(data))
result = read_jsonl_strip("/home/aliredaq/Desktop/DeepAgent/ICSE/src/tester/result_v2_openai-4o_new_zeroshot.jsonl", "file_path")
result = set(result)
print(datas - result)