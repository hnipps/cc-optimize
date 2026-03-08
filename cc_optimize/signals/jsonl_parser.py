from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    tool_use_id: str
    tool_name: str
    input_data: dict
    output: str | None
    is_error: bool
    timestamp_index: int


@dataclass
class AssistantBlock:
    text: str
    tool_calls: list[ToolCall]
    index: int


@dataclass
class ParsedSession:
    assistant_blocks: list[AssistantBlock]
    all_tool_calls: list[ToolCall]
    total_input_tokens: int
    total_output_tokens: int
    raw_events: list[dict]


_VALID_TYPES = {"system", "assistant", "user", "result"}


def parse(jsonl_path: Path) -> ParsedSession:
    """Parse a JSONL session file into a ParsedSession."""
    raw_events: list[dict] = []
    lines = jsonl_path.read_text().splitlines()

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON on line %d", line_no)
            continue
        raw_events.append(event)

    # Filter to valid event types
    events = [e for e in raw_events if e.get("type") in _VALID_TYPES]

    # First pass: collect tool_use entries from assistant events
    # and tool_result entries from user events
    tool_use_map: dict[str, ToolCall] = {}
    all_tool_calls: list[ToolCall] = []
    timestamp_index = 0

    # Build assistant blocks: contiguous assistant events before next user/end
    assistant_blocks: list[AssistantBlock] = []
    current_texts: list[str] = []
    current_tool_calls: list[ToolCall] = []
    in_assistant_turn = False
    block_index = 0

    for event in events:
        etype = event.get("type")

        if etype == "assistant":
            in_assistant_turn = True
            content = event.get("content", [])
            if isinstance(content, str):
                current_texts.append(content)
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_val = block.get("text", "")
                    if text_val:
                        current_texts.append(text_val)
                elif btype == "tool_use":
                    tc = ToolCall(
                        tool_use_id=block.get("id", ""),
                        tool_name=block.get("name", ""),
                        input_data=block.get("input", {}),
                        output=None,
                        is_error=False,
                        timestamp_index=timestamp_index,
                    )
                    timestamp_index += 1
                    tool_use_map[tc.tool_use_id] = tc
                    current_tool_calls.append(tc)
                    all_tool_calls.append(tc)

        elif etype == "user":
            # Flush current assistant turn if we were in one
            if in_assistant_turn:
                assistant_blocks.append(
                    AssistantBlock(
                        text="\n".join(current_texts),
                        tool_calls=current_tool_calls,
                        index=block_index,
                    )
                )
                block_index += 1
                current_texts = []
                current_tool_calls = []
                in_assistant_turn = False

            # Process tool_result content blocks
            content = event.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        tuid = block.get("tool_use_id", "")
                        if tuid in tool_use_map:
                            tc = tool_use_map[tuid]
                            tc.output = block.get("content", block.get("output", ""))
                            tc.is_error = bool(block.get("is_error", False))

        elif etype == "result":
            # Flush any pending assistant turn
            if in_assistant_turn:
                assistant_blocks.append(
                    AssistantBlock(
                        text="\n".join(current_texts),
                        tool_calls=current_tool_calls,
                        index=block_index,
                    )
                )
                block_index += 1
                current_texts = []
                current_tool_calls = []
                in_assistant_turn = False

    # Flush final assistant turn if still open
    if in_assistant_turn:
        assistant_blocks.append(
            AssistantBlock(
                text="\n".join(current_texts),
                tool_calls=current_tool_calls,
                index=block_index,
            )
        )

    # Extract token usage from result event
    result_events = [e for e in events if e.get("type") == "result"]
    if result_events:
        result_event = result_events[-1]
        usage = result_event.get("usage", {})
        total_input_tokens = usage.get("input_tokens", 0)
        total_output_tokens = usage.get("output_tokens", 0)
    else:
        logger.warning("No result event found in %s; token counts set to 0", jsonl_path)
        total_input_tokens = 0
        total_output_tokens = 0

    return ParsedSession(
        assistant_blocks=assistant_blocks,
        all_tool_calls=all_tool_calls,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        raw_events=raw_events,
    )
