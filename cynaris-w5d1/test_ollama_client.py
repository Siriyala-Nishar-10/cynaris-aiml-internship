"""
W5D1: Tests for the Ollama client and comparison helpers
-----------------------------------------------------------
These run WITHOUT Ollama installed: a tiny fake HTTP server mimics the parts of
Ollama's API the client uses (/api/tags and /api/chat), including its 404 for a
missing model. This verifies the client's request building, response parsing,
timing maths, and error handling -- it does NOT test any real model's output.

Run: python -m unittest test_ollama_client -v
"""

import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import compare_models
from ollama_client import (
    OllamaConnectionError, OllamaError, OllamaModelNotFoundError, chat, list_models,
)


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Mimics Ollama's response shapes. Model name picks the behaviour."""

    last_request = None  # the most recent /api/chat JSON body, for assertions

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send_json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send_json(200, {"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:3b"}]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        FakeOllamaHandler.last_request = body

        if body["model"] == "missing":
            self._send_json(404, {"error": "model 'missing' not found"})
        elif body["model"] == "broken":
            self._send_json(500, {"error": "boom"})
        else:
            self._send_json(200, {
                "model": body["model"],
                "message": {"role": "assistant", "content": "  Hello from the fake model.  "},
                "done": True,
                "load_duration": 500_000_000,       # 0.5 s
                "prompt_eval_count": 12,
                "eval_count": 50,
                "eval_duration": 2_000_000_000,     # 2 s  -> 25 tok/s
            })


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_list_models(self):
        self.assertEqual(list_models(self.base_url), ["llama3.2:3b", "qwen2.5:3b"])

    def test_chat_parses_reply_and_timings(self):
        result = chat("llama3.2:3b", "hi", base_url=self.base_url)
        self.assertEqual(result.content, "Hello from the fake model.")  # whitespace stripped
        self.assertEqual(result.completion_tokens, 50)
        self.assertEqual(result.prompt_tokens, 12)
        self.assertAlmostEqual(result.tokens_per_sec, 25.0)   # 50 tokens / 2 s
        self.assertAlmostEqual(result.load_s, 0.5)            # ns -> s
        self.assertGreater(result.latency_s, 0)

    def test_system_prompt_sent_before_user_message(self):
        chat("llama3.2:3b", "What is 2+2?", system_prompt="Be terse.", base_url=self.base_url)
        messages = FakeOllamaHandler.last_request["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "Be terse."})
        self.assertEqual(messages[1], {"role": "user", "content": "What is 2+2?"})
        self.assertFalse(FakeOllamaHandler.last_request["stream"])

    def test_no_system_prompt_sends_only_user_message(self):
        chat("llama3.2:3b", "hi", base_url=self.base_url)
        messages = FakeOllamaHandler.last_request["messages"]
        self.assertEqual([m["role"] for m in messages], ["user"])

    def test_options_forwarded(self):
        opts = {"temperature": 0.2, "seed": 42}
        chat("llama3.2:3b", "hi", options=opts, base_url=self.base_url)
        self.assertEqual(FakeOllamaHandler.last_request["options"], opts)

    def test_missing_model_raises_with_pull_hint(self):
        with self.assertRaises(OllamaModelNotFoundError) as ctx:
            chat("missing", "hi", base_url=self.base_url)
        self.assertIn("ollama pull missing", str(ctx.exception))

    def test_server_error_raises_generic_error(self):
        with self.assertRaises(OllamaError) as ctx:
            chat("broken", "hi", base_url=self.base_url)
        self.assertNotIsInstance(ctx.exception, OllamaModelNotFoundError)

    def test_connection_refused_raises_connection_error(self):
        # Grab a free port, then close it so nothing is listening there.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            dead_port = s.getsockname()[1]
        with self.assertRaises(OllamaConnectionError):
            chat("llama3.2:3b", "hi", base_url=f"http://127.0.0.1:{dead_port}", timeout=2)
        with self.assertRaises(OllamaConnectionError):
            list_models(f"http://127.0.0.1:{dead_port}", timeout=2)


class CheckHelperTests(unittest.TestCase):
    def test_contains_all_is_case_insensitive(self):
        self.assertTrue(compare_models.contains_all("By JAWAHARLAL NEHRU, 1946", ["nehru", "1946"]))
        self.assertFalse(compare_models.contains_all("By Nehru, 1947", ["nehru", "1946"]))

    def test_contains_number_is_standalone(self):
        self.assertTrue(compare_models.contains_number("The speed is 80 km/h.", "80"))
        self.assertTrue(compare_models.contains_number("It works out to 80.", "80"))
        self.assertFalse(compare_models.contains_number("That would be 180 km/h.", "80"))
        self.assertFalse(compare_models.contains_number("Roughly 80.5 km/h.", "80"))
        self.assertFalse(compare_models.contains_number("About 1.80 units.", "80"))

    def test_extract_bullets_handles_common_markers(self):
        text = "Intro line\n- first\n* second\n\u2022 third\n1. fourth\n2) fifth\nnot a bullet"
        self.assertEqual(compare_models.extract_bullets(text),
                         ["first", "second", "third", "fourth", "fifth"])

    def test_exactly_three_short_bullets(self):
        good = "- Learns noise\n- Great on train data\n- Poor on new data"
        self.assertTrue(compare_models.exactly_three_short_bullets(good))
        self.assertFalse(compare_models.exactly_three_short_bullets(good + "\n- One too many"))
        long_bullet = "- " + " ".join(["word"] * 25)
        self.assertFalse(compare_models.exactly_three_short_bullets(
            f"- short\n- short\n{long_bullet}"))
        numbered = "1. Memorises noise\n2. High train score\n3. Low test score"
        self.assertTrue(compare_models.exactly_three_short_bullets(numbered))


class ReportTests(unittest.TestCase):
    def test_build_report_contains_expected_sections(self):
        def fake_rows(passed):
            return [{
                "question_id": q["id"], "prompt": q["prompt"], "check_desc": q["check_desc"],
                "check_passed": passed, "response": "line one\nline two",
                "completion_tokens": 10, "tokens_per_sec": 12.3, "latency_s": 1.0, "word_count": 4,
            } for q in compare_models.QUESTIONS]

        report = compare_models.build_report({"model-a": fake_rows(True), "model-b": fake_rows(False)})
        self.assertIn("## Summary", report)
        self.assertIn("## Full responses", report)
        self.assertIn("## My observations", report)
        self.assertIn("PASS", report)
        self.assertIn("FAIL", report)
        for q in compare_models.QUESTIONS:
            self.assertIn(q["id"], report)


if __name__ == "__main__":
    unittest.main()
