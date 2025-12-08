#!/usr/bin/env python3
"""Pre-commit hook to enforce test co-location.

Ensures that:
1. All test files (test_*.py) are located in src/tests/unit/ or src/tests/property/
2. No test files exist outside of these directories
"""

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).parent.parent
    src_dir = root / "src"
    allowed_test_dirs = {
        src_dir / "tests" / "unit",
        src_dir / "tests" / "property",
    }

    errors = []

    # Find all test files in the project
    for test_file in src_dir.rglob("test_*.py"):
        # Check if the test file is in one of the allowed directories
        if not any(test_file.is_relative_to(allowed) for allowed in allowed_test_dirs):
            relative_path = test_file.relative_to(root)
            allowed_str = " or ".join(
                str(p.relative_to(root)) + "/" for p in sorted(allowed_test_dirs)
            )
            errors.append(f"  {relative_path} -> should be in {allowed_str}")

    # Also check for tests outside src/ (e.g., in project root)
    for test_file in root.glob("test_*.py"):
        relative_path = test_file.relative_to(root)
        errors.append(f"  {relative_path} -> should be in src/tests/unit/ or src/tests/property/")

    # Check tests/ directory at root level
    root_tests = root / "tests"
    if root_tests.exists() and root_tests.is_dir():
        for test_file in root_tests.rglob("test_*.py"):
            relative_path = test_file.relative_to(root)
            errors.append(f"  {relative_path} -> should be in src/tests/unit/ or src/tests/property/")

    if errors:
        print("ERROR: Test files must be co-located in src/tests/unit/ or src/tests/property/")
        print()
        print("Found test files in wrong locations:")
        for error in errors:
            print(error)
        print()
        print("Please move these files to src/tests/unit/ or src/tests/property/")
        return 1

    # Ensure allowed directories exist (helpful on fresh clones)
    for allowed_dir in allowed_test_dirs:
        if not allowed_dir.exists():
            print(f"WARNING: {allowed_dir.relative_to(root)}/ directory does not exist")
            print("Creating it now...")
            allowed_dir.mkdir(parents=True, exist_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
