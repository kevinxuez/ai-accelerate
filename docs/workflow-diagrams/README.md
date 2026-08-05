# CaseFile workflow charts

This folder contains eleven high-resolution PNG flowcharts derived from the current-state
CaseFile agent workflow descriptions.

## Color key

- Navy: entry point
- Blue: system or tool step
- Gold: decision or confirmation point
- Purple: model-assisted step, write, or special policy gate
- Red: denial, rejection, or blocked path
- Green: returned result or completed terminal state

The editable Mermaid sources are in `src/`. The shared rendering settings are in
`mermaid-config.json`.

## Regenerate the PNG set

Run from the repository root with Google Chrome installed:

```sh
for source_file in docs/workflow-diagrams/src/*.mmd; do
  image_file="docs/workflow-diagrams/png/$(basename "${source_file%.mmd}").png"
  PUPPETEER_EXECUTABLE_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
    PUPPETEER_SKIP_DOWNLOAD=true \
    npx -y @mermaid-js/mermaid-cli \
    -i "$source_file" \
    -o "$image_file" \
    -c docs/workflow-diagrams/mermaid-config.json \
    -b '#F8FAFC' \
    -s 2
done
```
