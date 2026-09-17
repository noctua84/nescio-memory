import json
import yaml
from main import app

def export():
    spec = app.openapi()

    # Export as JSON
    with open('openapi.json', 'w') as f:
        json.dump(spec, f, indent=2)

    # Export as YAML
    with open('openapi.yaml', 'w') as f:
        yaml.dump(spec, f, sort_keys=False)


    print("Exported openapi.json and openapi.yaml")


if __name__ == '__main__':
    export()