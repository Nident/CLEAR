from __future__ import annotations

import time
from typing import Any, Callable, Optional

from utils.dataloader.message_saver import load_agent_messages, save_all_field


def run_with_retry(
    fn: Callable[[], Any],
    is_ok: Callable[[Any], bool],
    retries: int,
    delay_sec: float,
) -> Any:
    last: Any = None
    for attempt in range(retries + 1):
        last = fn()
        if is_ok(last):
            return last
        if attempt < retries:
            time.sleep(delay_sec)
    return last


def persist_all_field_safe(index: int, messages_dir: str, model_name: Optional[str]) -> None:
    try:
        bundle = load_agent_messages(index=index, messages_dir=messages_dir)
        save_all_field(
            index=index,
            messages_dir=messages_dir,
            messages=bundle,
            model_name=model_name,
            run_id=None,
            parent_run_id=None,
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        return
