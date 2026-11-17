"""Static source-contract tests; do not validate archived numerical results."""
from pathlib import Path
import ast
import unittest

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "scripts" / "step_41c_frozen_potsdam_transfer.py"

class TestStep41CSource(unittest.TestCase):
    def test_bicubic_reference_with_align_corners_false(self):
        module = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "registration_reference"]
        self.assertEqual(len(functions), 1)
        calls = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "interpolate"]
        self.assertEqual(len(calls), 1)
        kwargs = {k.arg: k.value for k in calls[0].keywords}
        self.assertEqual(ast.literal_eval(kwargs["mode"]), "bicubic")
        self.assertIs(ast.literal_eval(kwargs["align_corners"]), False)
        self.assertEqual(ast.literal_eval(kwargs["size"]), (96,96))

if __name__ == "__main__":
    unittest.main()
