import glob
import os
import subprocess
import sys
import unittest

from helpers import ROOT


class ExampleTests(unittest.TestCase):
    """Every program in examples/ must run to completion without errors."""

    def test_examples_run(self):
        files = sorted(glob.glob(os.path.join(ROOT, "examples", "*.sn")))
        self.assertGreaterEqual(len(files), 10)
        for path in files:
            with self.subTest(example=os.path.basename(path)):
                r = subprocess.run([sys.executable, "-m", "sn", path], cwd=ROOT,
                                   capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(r.stdout.strip())


class CliTests(unittest.TestCase):
    def sn(self, *args):
        return subprocess.run([sys.executable, "-m", "sn", *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)

    def test_version(self):
        self.assertIn("S#N", self.sn("--version").stdout)

    def test_eval(self):
        self.assertEqual(self.sn("eval", "print(6 * 7)").stdout.strip(), "42")

    def test_check_ok_and_error(self):
        self.assertEqual(self.sn("check", "examples/01_hello.sn").returncode, 0)
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".sn", delete=False) as f:
            f.write("let = 1\n")
        r = self.sn("check", f.name)
        os.unlink(f.name)
        self.assertEqual(r.returncode, 1)
        self.assertIn("SyntaxError", r.stderr)

    def test_runtime_error_exit_code_and_message(self):
        r = self.sn("eval", "let x = 1\nprint(x / 0)")
        self.assertEqual(r.returncode, 1)
        self.assertIn("division by zero", r.stderr)
        self.assertIn("-->", r.stderr)

    def test_exit_code_from_program(self):
        self.assertEqual(self.sn("eval", "exit(3)").returncode, 3)

    def test_program_arguments(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".sn", delete=False) as f:
            f.write("print(args)\n")
        r = self.sn(f.name, "a", "b")
        os.unlink(f.name)
        self.assertEqual(r.stdout.strip(), '["a", "b"]')


if __name__ == "__main__":
    unittest.main()
