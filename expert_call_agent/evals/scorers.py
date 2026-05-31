"""Pure scoring functions for Weave evaluations.

Each scorer takes dataset columns (by name) plus the model `output` dict and
returns a dict of metrics. No I/O, so they are trivially checkable.
"""


def flag_type_match(expected_flag: str, output: dict) -> dict:
    """Did the agent assign the expected flag tier?"""
    got = (output or {}).get("flag_type")
    return {"correct": got == expected_flag}


def produced_question(output: dict) -> dict:
    """Did the agent actually produce a non-empty question?"""
    content = (output or {}).get("content", "")
    return {"nonempty": bool(content and content.strip())}
