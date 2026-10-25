import json
import yaml
from app.main import app

# release-please rewrites info.version in both files on every release, and its
# GenericJson/GenericYaml updaters re-serialize the whole document rather than
# patching one line. openapi.json must match theirs byte for byte, or the CI
# drift check fails on the release commit itself; their JSON differs from ours
# in one way, verified against an actual release PR:
#   - integral floats are written as integers (50.0 -> 50)
# openapi.yaml only needs the same parsed content: the drift check compares it
# with yaml.safe_load, because their YAML emitter folds long strings into `>-`
# blocks that PyYAML cannot reproduce. We still indent sequence items two
# columns under their parent key, as they do, to keep release diffs small.
# If they ever change serializer, the drift check will catch it immediately.
_NUMERIC_SCHEMA_KEYS = frozenset(
    {"maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum", "multipleOf"}
)


class _IndentedDumper(yaml.Dumper):
    """yaml.Dumper, but with sequences indented under their parent mapping."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _normalize_numeric_constraints(node):
    if isinstance(node, dict):
        return {
            key: (
                int(value)
                if key in _NUMERIC_SCHEMA_KEYS
                and isinstance(value, float)
                and value.is_integer()
                else _normalize_numeric_constraints(value)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_normalize_numeric_constraints(item) for item in node]
    return node


def export():
    spec = _normalize_numeric_constraints(app.openapi())

    # Export as JSON
    with open('openapi.json', 'w') as f:
        json.dump(spec, f, indent=2)

    # Export as YAML
    with open('openapi.yaml', 'w') as f:
        yaml.dump(spec, f, Dumper=_IndentedDumper, sort_keys=False)


    print("Exported openapi.json and openapi.yaml")


if __name__ == '__main__':
    export()
