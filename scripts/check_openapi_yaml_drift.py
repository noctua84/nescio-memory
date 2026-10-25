#!/usr/bin/env python
"""Check openapi.yaml for drift by comparing parsed content, not bytes.

release-please's GenericYaml updater re-serializes with different YAML
formatting (folded >- scalars for long strings vs plain/single-quoted),
so a byte diff would fail on every release PR even though content is identical.
"""

import subprocess
import sys
import yaml


def main() -> int:
    try:
        # Get committed version
        committed_yaml = subprocess.check_output(
            ["git", "show", "HEAD:openapi.yaml"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        committed = yaml.safe_load(committed_yaml)
    except subprocess.CalledProcessError:
        print("::error::Could not retrieve committed openapi.yaml")
        return 1

    # Get freshly generated version
    with open("openapi.yaml", encoding="utf-8") as f:
        generated_yaml = f.read()
    generated = yaml.safe_load(generated_yaml)

    # Compare parsed content
    if committed == generated:
        # Check if bytes differ (formatting only)
        if committed_yaml != generated_yaml:
            print("ℹ️  openapi.yaml content matches but formatting differs (expected on release-please PRs)")
        return 0

    # Content mismatch
    print("::error::openapi.yaml content has drifted!")
    print("Run 'uv run python export_openapi.py' locally and commit the result.")
    subprocess.run(["git", "diff", "openapi.yaml"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
