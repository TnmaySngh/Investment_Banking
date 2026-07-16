# =============================================================================
# MODULE: generate_tree
# PURPOSE:
#     Produce a clean, deterministic, human-readable directory tree for any
#     project folder. This version filters out noisy internal folders such as
#     .git/objects, __pycache__, and hidden system folders. Output includes
#     file sizes and last-modified timestamps with short timezone abbreviations
#     (EST/EDT). Results are written to logs/folder_tree.log for architectural
#     validation, onboarding, and debugging import issues.
#
# USAGE:
#     Option 1 — Default (project root):
#         python generate_tree.py
#
#     Option 2 — Relative folder:
#         python generate_tree.py src
#         python generate_tree.py src/engines
#
#     Option 3 — Absolute folder:
#         python generate_tree.py C:\Projects\AIML\rani-predict-direction\src
#         python generate_tree.py /Users/ahalya/Projects/src
#
# MODULE SUMMARY:
#     • Recursively walks directory structure
#     • Filters out noisy folders (e.g., .git/objects, __pycache__)
#     • Prints file sizes and last-modified timestamps
#     • Writes deterministic output to logs/folder_tree.log
#     • Supports onboarding, CI/CD, and architecture audits
#
# ARCHITECTURAL ROLE:
#     Developer-facing utility used to validate canonical project structure,
#     detect missing __init__.py files, and ensure folder scaffolding remains
#     stable across refactors. Not used at runtime; supports engineering
#     governance and architectural hygiene.
#
# PROVIDER REGISTRY MAP:
#     - Upstream Providers:
#         • OS filesystem
#     - Downstream Consumers:
#         • Developers
#         • CI/CD validation
#         • Architecture audits
#
# FALLBACK CHAINS:
#     - If folder does not exist → write error line and continue
#     - If path is not a directory → write error line and continue
#     - If logs/ folder missing → auto-created deterministically
#
# EXECUTION FLOW:
#     1. Parse input path (default = ".")
#     2. Ensure logs/ folder exists
#     3. Generate folder tree recursively
#     4. Write output to logs/folder_tree.log
#     5. Print console confirmation
#
# INPUT / OUTPUT:
#     INPUT:
#         - Optional folder path (absolute or relative)
#     OUTPUT:
#         - folder_tree.log containing full directory tree
#
# DEPENDENCIES:
#     - os for filesystem traversal
#     - sys for argument parsing
#     - datetime for timestamping
#     - time for timezone-aware formatting
#
# EXTENSION POINTS:
#     - Add last-modified timestamps
#     - Add file counts per folder
#     - Add colorized console output
#     - Add filtering (e.g., only .py files)
#
# TESTING STRATEGY:
#     - Patch os.listdir to simulate folder structures
#     - Patch os.path.getsize for deterministic size output
#     - Validate empty-file detection
#     - Validate nested folder indentation
#     - Validate skip-folder filtering
#     - Validate timestamp formatting
# =============================================================================

import os  # filesystem operations
import sys  # argument parsing
from datetime import datetime  # timestamp formatting
import time  # timezone-aware timestamp conversion


# =============================================================================
# CONSTANT: SKIP_FOLDERS
# PURPOSE:
#     Define folders that should be excluded from the directory tree output.
#     Prevents noise from .git/objects, __pycache__, and other internal folders.
# =============================================================================
SKIP_FOLDERS = {
    ".git",              # skip entire .git folder
    "__pycache__",       # skip Python cache
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
}


# =============================================================================
# FUNCTION: format_size
# PURPOSE:
#     Convert raw byte size into a human-readable label. Marks zero-byte files
#     explicitly as "(empty)" for debugging missing or placeholder modules.
# =============================================================================
def format_size(bytes_value: int) -> str:
    """Return human-readable file size label."""  # inline comment
    if bytes_value == 0:  # check for empty file
        return "(empty)"  # explicit empty marker
    kb = bytes_value / 1024  # convert bytes to kilobytes
    return f"{kb:.1f} KB"  # formatted size label


# =============================================================================
# FUNCTION: format_timestamp
# PURPOSE:
#     Convert a UNIX timestamp into ISO 8601 format with short timezone
#     abbreviation (EST/EDT). Ensures consistent output across OS locales.
# =============================================================================
def format_timestamp(ts: float) -> str:
    """Return ISO 8601 timestamp with short timezone abbreviation."""  # inline comment

    local_time = time.localtime(ts)  # convert to local time
    formatted = time.strftime("%Y-%m-%d %H:%M:%S %Z", local_time)  # base formatting

    # Normalize long timezone names to short abbreviations
    replacements = {
        "Eastern Standard Time": "EST",
        "Eastern Daylight Time": "EDT",
        "Eastern Time": "ET",
    }

    for long, short in replacements.items():  # iterate through replacements
        if long in formatted:  # detect long form
            formatted = formatted.replace(long, short)  # replace with short form

    return formatted  # return normalized timestamp


# =============================================================================
# FUNCTION: generate_tree
# PURPOSE:
#     Recursively walk the directory structure starting at start_path and build
#     an indented tree representation including file sizes and timestamps.
#     Skips noisy internal folders for clarity.
# =============================================================================
def generate_tree(start_path: str, prefix: str = "", output_lines=None):
    """Recursively builds the folder tree structure with file sizes + timestamps."""  # inline comment

    if output_lines is None:  # initialize list on first call
        output_lines = []  # create new list for output lines

    try:
        items = sorted(os.listdir(start_path))  # deterministic listing
    except FileNotFoundError:
        output_lines.append(f"[ERROR] Folder not found: {start_path}")  # error line
        return output_lines  # return early
    except NotADirectoryError:
        output_lines.append(f"[ERROR] Not a directory: {start_path}")  # error line
        return output_lines  # return early

    for index, item in enumerate(items):  # iterate through directory items

        # Skip noisy folders
        if item in SKIP_FOLDERS:  # skip known noisy folders
            continue  # skip entry

        # Skip hidden folders except .env
        if item.startswith(".") and item != ".env":  # skip hidden folders
            continue  # skip entry

        path = os.path.join(start_path, item)  # build full path
        connector = "└── " if index == len(items) - 1 else "├── "  # tree connector
        timestamp = format_timestamp(os.path.getmtime(path))  # last-modified timestamp

        if os.path.isdir(path):  # handle folder
            output_lines.append(prefix + connector + f"{item}/  ({timestamp})")  # folder line
            extension = "    " if index == len(items) - 1 else "│   "  # indentation
            generate_tree(path, prefix + extension, output_lines)  # recurse
        else:
            size = os.path.getsize(path)  # file size in bytes
            size_label = format_size(size)  # readable label
            output_lines.append(
                prefix + connector + f"{item}  [{size_label}]  ({timestamp})"
            )  # file line

    return output_lines  # return accumulated lines


# =============================================================================
# ENTRY POINT
# PURPOSE:
#     Allow direct execution from command line to generate a folder tree for
#     any path. Writes output to logs/folder_tree.log and prints confirmation.
# =============================================================================
if __name__ == "__main__":  # allow module to be executed directly

    raw_path = sys.argv[1] if len(sys.argv) > 1 else "."  # read input path or default
    ENTRY_POINT = os.path.abspath(raw_path)  # normalize to absolute path

    LOG_FOLDER = "logs"  # canonical log folder
    os.makedirs(LOG_FOLDER, exist_ok=True)  # ensure logs folder exists

    LOG_FILE = os.path.join(LOG_FOLDER, "folder_tree.log")  # canonical log file path

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # timestamp

    header = [
        "==============================================================",
        f"  Project Folder Tree",
        f"  ENTRY POINT: {ENTRY_POINT}",
        f"  Execution Time: {timestamp}",
        "==============================================================",
        "",
    ]  # header lines

    tree_output = generate_tree(ENTRY_POINT)  # generate folder tree

    with open(LOG_FILE, "w", encoding="utf-8") as f:  # open log file
        for line in header + tree_output:  # iterate through header + tree
            f.write(line + "\n")  # write each line

    print(f"Folder tree written to: {LOG_FILE}")  # console confirmation
    print(f"Scanned folder: {ENTRY_POINT}")  # console confirmation
