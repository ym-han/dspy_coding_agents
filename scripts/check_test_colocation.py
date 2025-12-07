#!/usr/bin/env python3
"""Pre-commit hook to enforce test co-location.

Ensures that:
1. All test files (test_*.py) are located in src/tests/unit/
2. No test files exist outside of this directory
"""

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).parent.parent
    src_dir = root / "src"
    allowed_test_dir = src_dir / "tests" / "unit"

    errors = []

    # Find all test files in the project
    for test_file in src_dir.rglob("test_*.py"):
        # Check if the test file is in the allowed directory
        try:
            test_file.relative_to(allowed_test_dir)
        except ValueError:
            # File is not under allowed_test_dir
            relative_path = test_file.relative_to(root)
            errors.append(
                f"  {relative_path} -> should be in src/tests/unit/"
            )

    # Also check for tests outside src/ (e.g., in project root)
    for test_file in root.glob("test_*.py"):
        relative_path = test_file.relative_to(root)
        errors.append(
            f"  {relative_path} -> should be in src/tests/unit/"
        )

    # Check tests/ directory at root level
    root_tests = root / "tests"
    if root_tests.exists() and root_tests.is_dir():
        for test_file in root_tests.rglob("test_*.py"):
            relative_path = test_file.relative_to(root)
            errors.append(
                f"  {relative_path} -> should be in src/tests/unit/"
            )

    if errors:
        print("ERROR: Test files must be co-located in src/tests/unit/")
        print()
        print("Found test files in wrong locations:")
        for error in errors:
            print(error)
        print()
        print("Please move these files to src/tests/unit/")
        return 1

    # Verify the test directory exists
    if not allowed_test_dir.exists():
        print("WARNING: src/tests/unit/ directory does not exist")
        print("Creating it now...")
        allowed_test_dir.mkdir(parents=True, exist_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
