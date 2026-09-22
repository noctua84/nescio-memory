import json
import yaml
from app.main import app

# release-please rewrites info.version in both files on every release, and its
# GenericJson/GenericYaml updaters re-serialize the whole document rather than
# patching one line. Our own output therefore has to match theirs byte for byte,
# or the CI drift check fails on the release commit itself. Two differences
# matter, both verified against an actual release PR:
#   - their YAML indents sequence items two columns under their parent key
#   - their JSON writes integral floats as integers (50.0 -> 50)
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
