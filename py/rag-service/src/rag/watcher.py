"""File-system watcher for live re-indexing of local resources."""

from __future__ import annotations

import threading
from pathlib import Path

from libs.logger import logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler


class FileSystemHandler(FileSystemEventHandler):
    """Handler for file system events."""

    def __init__(self: FileSystemHandler, directory: Path) -> None:
        """Initialise the handler."""
        self.directory = directory

    def on_modified(self: FileSystemHandler, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory and not str(event.src_path).endswith(".tmp"):
            self.handle_file_change(Path(str(event.src_path)))

    def on_created(self: FileSystemHandler, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory and not str(event.src_path).endswith(".tmp"):
            self.handle_file_change(Path(str(event.src_path)))

    def handle_file_change(self: FileSystemHandler, file_path: Path) -> None:
        """Handle changes to a file."""
        # Defer import to avoid circular dependency at module load time.
        from rag.engine import BATCH_PROCESSING_DELAY, file_last_modified, update_index_for_file  # noqa: PLC0415

        current_time = __import__("time").time()

        abs_file_path = file_path
        if not Path(abs_file_path).is_absolute():
            abs_file_path = Path(self.directory, file_path)

        # Check if the file was recently processed
        if (
            abs_file_path in file_last_modified
            and current_time - file_last_modified[abs_file_path] < BATCH_PROCESSING_DELAY
        ):
            return

        file_last_modified[abs_file_path] = current_time
        logger.debug("Scheduling re-index for changed file: %s", abs_file_path)
        threading.Thread(target=update_index_for_file, args=(self.directory, abs_file_path)).start()

