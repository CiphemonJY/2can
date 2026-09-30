"""Regression test: a run-on source must not push the claim out of the FactCG prompt.

Loads only the FactCG tokenizer, not the models. Run: python -m unittest discover tests
"""
import types
import unittest

from transformers import AutoTokenizer

from two_can import server


class FactCGPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = AutoTokenizer.from_pretrained("yaxili96/FactCG-DeBERTa-v3-Large")
        server.ENG["fc"] = types.SimpleNamespace(tok=tok, maxlen=2048)
        cls.tok = tok

    def test_run_on_source_is_chunked(self):
        src = " ".join("word%d" % i for i in range(2300))          # no sentence breaks at all
        chunks = server.fc_chunks(src)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.split()) <= server.FC_CHUNK_WORDS for c in chunks))

    def test_claim_survives_truncation(self):
        chunk = " ".join(["alpha beta gamma delta"] * 1500)           # far over 2048 tokens as one chunk
        claim = "Penguins are the national animal of Brazil today."
        p = server.fc_prompt(chunk, claim)
        self.assertLessEqual(len(self.tok.encode(p)), 2048)
        self.assertIn(claim, p)

    def test_normal_sentences_untouched(self):
        src = "The bridge opened in 1937.  It is  red.\nIt spans the strait."
        self.assertEqual(server.fc_chunks(src), ["The bridge opened in 1937.\nIt is  red.\nIt spans the strait."])


if __name__ == "__main__":
    unittest.main()
