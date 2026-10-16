"""
W5D2: Tests for prompt templates and the persona/technique comparison logic
-------------------------------------------------------------------------------
Run: python -m unittest test_prompt_engineering -v

Like W5D1's tests, these use a fake HTTP server so they run without Ollama
installed. They verify template rendering and the label/JSON extraction
helpers -- not any real model's output.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import prompting_techniques
import system_prompt_experiment
from prompt_templates import FewShotPromptTemplate, PromptTemplate, chain_of_thought


class TestPromptTemplate(unittest.TestCase):
    def test_from_template_detects_variables(self):
        t = PromptTemplate.from_template(
            "Explain {topic} to a {audience}."
        )
        self.assertEqual(
            t.input_variables,
            ["audience", "topic"]
        )

    def test_format_fills_placeholders(self):
        t = PromptTemplate.from_template(
            "Explain {topic} to a {audience}."
        )
        self.assertEqual(
            t.format(
                topic="overfitting",
                audience="beginner"
            ),
            "Explain overfitting to a beginner."
        )

    def test_format_raises_on_missing_variable(self):
        t = PromptTemplate.from_template(
            "Explain {topic}."
        )

        with self.assertRaises(ValueError) as ctx:
            t.format()

        self.assertIn("topic", str(ctx.exception))

    def test_template_with_no_variables(self):
        t = PromptTemplate.from_template("Say hello.")

        self.assertEqual(t.input_variables, [])
        self.assertEqual(t.format(), "Say hello.")


class TestFewShotPromptTemplate(unittest.TestCase):
    def setUp(self):
        self.template = FewShotPromptTemplate(
            prefix="Classify sentiment.",
            example_template=PromptTemplate.from_template(
                'Text: "{text}"\nSentiment: {label}'
            ),
            examples=[
                {
                    "text": "Great!",
                    "label": "POSITIVE"
                },
                {
                    "text": "Terrible.",
                    "label": "NEGATIVE"
                },
            ],
            suffix='Text: "{text}"\nSentiment:',
        )

    def test_includes_prefix_all_examples_and_suffix(self):
        result = self.template.format(text="It was okay.")

        self.assertIn(
            "Classify sentiment.",
            result
        )

        self.assertIn(
            'Text: "Great!"\nSentiment: POSITIVE',
            result
        )

        self.assertIn(
            'Text: "Terrible."\nSentiment: NEGATIVE',
            result
        )

        self.assertTrue(
            result.strip().endswith(
                'Text: "It was okay."\nSentiment:'
            )
        )

    def test_example_order_is_preserved(self):
        result = self.template.format(text="x")

        self.assertLess(
            result.index("Great!"),
            result.index("Terrible.")
        )


class TestChainOfThought(unittest.TestCase):
    def test_appends_trigger_phrase(self):
        result = chain_of_thought("What is 2+2?")

        self.assertTrue(
            result.startswith("What is 2+2?")
        )

        self.assertIn(
            "Let's think step by step.",
            result
        )


class TestExtractLabel(unittest.TestCase):
    def test_extracts_negative(self):
        self.assertEqual(
            prompting_techniques.extract_label(
                "Sentiment: NEGATIVE"
            ),
            "NEGATIVE"
        )

    def test_extracts_positive_lowercase_response(self):
        self.assertEqual(
            prompting_techniques.extract_label(
                "this is positive"
            ),
            "POSITIVE"
        )

    def test_unclear_when_neither_present(self):
        self.assertEqual(
            prompting_techniques.extract_label(
                "I'm not sure about this one."
            ),
            "UNCLEAR"
        )

    def test_picks_first_match_when_both_present(self):
        # Documents a real limitation: a response that mentions both words
        # only has the FIRST one extracted, which may not be the model's
        # actual final answer if it reasons through both possibilities.

        text = "It's not POSITIVE, it's actually NEGATIVE."

        self.assertEqual(
            prompting_techniques.extract_label(text),
            "POSITIVE"
        )


class TestLooksLikeJson(unittest.TestCase):
    def test_valid_json_object(self):
        self.assertTrue(
            system_prompt_experiment.looks_like_json(
                '{"a": 1}'
            )
        )

    def test_valid_json_with_surrounding_whitespace(self):
        self.assertTrue(
            system_prompt_experiment.looks_like_json(
                '  {"a": 1}  \n'
            )
        )

    def test_invalid_json_with_markdown_fence(self):
        self.assertFalse(
            system_prompt_experiment.looks_like_json(
                '```json\n{"a": 1}\n```'
            )
        )

    def test_plain_text_is_not_json(self):
        self.assertFalse(
            system_prompt_experiment.looks_like_json(
                "Sure, here's my answer."
            )
        )


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """
    Minimal fake server:

    /api/tags lists both models.

    /api/chat echoes a fixed reply so the calling scripts can be
    exercised end-to-end without a real Ollama installation.
    """

    def log_message(self, *args):
        pass

    def _send_json(self, status, body):
        raw = json.dumps(body).encode()

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json"
        )
        self.send_header(
            "Content-Length",
            str(len(raw))
        )
        self.end_headers()

        self.wfile.write(raw)

    def do_GET(self):
        self._send_json(
            200,
            {
                "models": [
                    {
                        "name": "llama3.2:3b"
                    },
                    {
                        "name": "qwen2.5:3b"
                    }
                ]
            }
        )

    def do_POST(self):
        length = int(
            self.headers.get(
                "Content-Length",
                0
            )
        )

        # Validate that the request body is valid JSON.
        json.loads(
            self.rfile.read(length)
        )

        self._send_json(
            200,
            {
                "model": "llama3.2:3b",
                "message": {
                    "role": "assistant",
                    "content": "NEGATIVE"
                },
                "done": True,
                "load_duration": 0,
                "prompt_eval_count": 5,
                "eval_count": 3,
                "eval_duration": 300_000_000,
            }
        )


class TestScriptsRunEndToEnd(unittest.TestCase):
    """
    Smoke-tests both scripts' main() against a fake server,
    without needing real Ollama installed.
    """

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            FakeOllamaHandler
        )

        cls.port = cls.server.server_address[1]

        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True
        )

        cls.thread.start()

        import ollama_client

        cls._orig_base_url = (
            ollama_client.DEFAULT_BASE_URL
        )

        ollama_client.DEFAULT_BASE_URL = (
            f"http://127.0.0.1:{cls.port}"
        )

    @classmethod
    def tearDownClass(cls):
        import ollama_client

        ollama_client.DEFAULT_BASE_URL = (
            cls._orig_base_url
        )

        cls.server.shutdown()
        cls.server.server_close()

    def test_system_prompt_experiment_runs_and_writes_output(self):
        import sys

        old_argv = sys.argv

        # Windows-safe relative output path.
        output_file = "test_sp_results.json"

        sys.argv = [
            "system_prompt_experiment.py",
            "--output",
            output_file
        ]

        try:
            exit_code = system_prompt_experiment.main()
        finally:
            sys.argv = old_argv

        self.assertEqual(
            exit_code,
            0
        )

        with open(
            output_file,
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        self.assertEqual(
            len(data["results"]),
            4
        )


    def test_prompting_techniques_runs_and_writes_output(self):
        import sys

        old_argv = sys.argv

        # Windows-safe relative output path.
        output_file = "test_pt_results.json"

        sys.argv = [
            "prompting_techniques.py",
            "--output",
            output_file
        ]

        try:
            exit_code = prompting_techniques.main()
        finally:
            sys.argv = old_argv

        self.assertEqual(
            exit_code,
            0
        )

        with open(
            output_file,
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        self.assertEqual(
            len(data["results"]),
            3
        )

        # Fake server always returns "NEGATIVE",
        # so every technique should extract it correctly.
        self.assertTrue(
            all(
                r["correct"]
                for r in data["results"]
            )
        )


if __name__ == "__main__":
    unittest.main()