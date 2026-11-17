"""Non-executing archive checks; NOT scientific reproduction tests."""
from pathlib import Path
import ast
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ArchiveIntegrityTests(unittest.TestCase):
    def test_eight_archival_scripts_parse(self):
        scripts = sorted((ROOT / "scripts").glob("step_*.py"))
        self.assertEqual(len(scripts), 8)
        for path in scripts:
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_notebooks_have_no_saved_outputs(self):
        notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 2)
        for path in notebooks:
            with self.subTest(file=path.name):
                nb = json.loads(path.read_text(encoding="utf-8"))
                for cell in nb.get("cells", []):
                    if cell.get("cell_type") == "code":
                        self.assertEqual(cell.get("outputs", []), [])
                        self.assertIsNone(cell.get("execution_count"))

    def test_expected_metadata_exists(self):
        for name in ("README.md", "AUTHORS.md", "CITATION.cff", "docs/PUBLIC_RELEASE_CHECKLIST.md"):
            self.assertTrue((ROOT / name).is_file(), name)

if __name__ == "__main__":
    unittest.main()
