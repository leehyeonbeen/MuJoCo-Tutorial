# Contributing to MuJoCo Tutorial 🤝

Thanks for your interest in contributing! This repo aims to be the best starting point for learning MuJoCo.

## Ways to Contribute

1. **Add a new example** — standalone scripts in `examples/`
2. **Improve tutorials** — fix errors, add explanations, update for new MuJoCo versions
3. **Fix bugs** — if an example doesn't work, open an issue or PR
4. **Add translations** — README in other languages

## Adding a New Example

1. Fork and clone the repo
2. Create `examples/your_example.py`
3. Follow the existing style:
   - Docstring at the top explaining what the example demonstrates
   - Inline MJCF XML (no external files needed)
   - Use `mujoco.viewer.launch_passive()` for interactive viewing
   - Keep it under 150 lines
4. Add a row to the examples table in `README.md`
5. Submit a PR

### Example Template

```python
"""Short description — what control/physics concept this demonstrates."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <!-- Your model here -->
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 10.0:
            # Your controller here
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
```

## Adding/Editing Tutorials

- Notebooks are in `tutorial/` and numbered sequentially
- Each notebook should have:
  - A Colab badge at the top
  - A `!pip install -q mujoco mediapy` cell
  - No embedded outputs (keeps files small)
  - Clear markdown explanations between code cells

## Code Style

- Python 3.10+
- No type annotations required (keep examples simple)
- Use `numpy` style (no one-letter variables except standard math: `x`, `q`, `F`)
- Prefer inline MJCF over external XML files

## Running Tests

```bash
python scripts/test_examples.py
```

This runs each example headlessly for a few steps to verify it doesn't crash.

## Questions?

Open an issue or start a discussion. We're happy to help!
