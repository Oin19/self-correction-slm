"""Tests for resumable DPO preference-pair collection (JSONL checkpointing)."""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src.training import dpo as dpo_mod


class _FakeDataset:
    @staticmethod
    def from_list(items):
        return list(items)


def _hist(*statuses):
    return [{"turn": i + 1, "code": f"code_{s}_{i}", "result": {"status": s}}
            for i, s in enumerate(statuses)]


PROBLEMS = [{"question": "q0"}, {"question": "q1"}, {"question": "q2"}, {"question": "q3"}]

HISTORY = {
    "q0": _hist("WA", "AC"),        # recovery -> pair
    "q1": _hist("CE", "RE", "WA"),  # all bad -> no pair
    "q2": _hist("AC"),              # turn-1 AC, no rejected -> no pair
}


def _parse(prob):
    return [] if prob["question"] == "q3" else ["assert True"]


class TestPreferencePairCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self.tmp.name, "pairs.jsonl")
        self.calls = []
        ds_stub = types.ModuleType("datasets")
        ds_stub.Dataset = _FakeDataset

        def _loop(model, tokenizer, problem, test_cases=None, K=3,
                  initial_code=None, max_gen_time=None):
            self.calls.append({"problem": problem, "K": K, "max_gen_time": max_gen_time})
            return HISTORY[problem]

        self.patches = [
            patch.dict(sys.modules, {"datasets": ds_stub}),
            patch.object(dpo_mod, "agentic_debug_loop", side_effect=_loop),
            patch.object(dpo_mod, "parse_apps_test_cases", side_effect=_parse),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _read_ckpt(self):
        entries = []
        with open(self.ckpt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # malformed line tolerated, matching production loader
        return entries

    def test_full_run_writes_checkpoint_and_pairs(self):
        pairs = dpo_mod.make_preference_pairs(PROBLEMS, None, None, K=3, seed=42,
                                              checkpoint_path=self.ckpt, max_gen_time=90)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt"], "q0")
        self.assertEqual(pairs[0]["chosen"], "code_AC_1")
        self.assertEqual(pairs[0]["rejected"], "code_WA_0")
        entries = self._read_ckpt()
        self.assertEqual([e["idx"] for e in entries], [0, 1, 2, 3])
        self.assertIsNone(entries[1]["pair"])
        # q3 has no executable tests: checkpointed, but debug loop never called
        self.assertEqual([c["problem"] for c in self.calls], ["q0", "q1", "q2"])
        self.assertTrue(all(c["max_gen_time"] == 90 for c in self.calls))
        self.assertTrue(all(c["K"] == 3 for c in self.calls))

    def test_resume_skips_all_processed_problems(self):
        dpo_mod.make_preference_pairs(PROBLEMS, None, None,
                                      checkpoint_path=self.ckpt, max_gen_time=90)
        self.calls.clear()
        pairs2 = dpo_mod.make_preference_pairs(PROBLEMS, None, None,
                                               checkpoint_path=self.ckpt, max_gen_time=90)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(pairs2), 1)

    def test_interrupted_run_resumes_only_missing_problems(self):
        dpo_mod.make_preference_pairs(PROBLEMS, None, None, checkpoint_path=self.ckpt)
        with open(self.ckpt, encoding="utf-8") as f:
            lines = f.readlines()
        with open(self.ckpt, "w", encoding="utf-8") as f:
            f.writelines(lines[:-2])  # simulate crash after idx 1
        self.calls.clear()
        pairs = dpo_mod.make_preference_pairs(PROBLEMS, None, None,
                                              checkpoint_path=self.ckpt, max_gen_time=90)
        self.assertEqual([c["problem"] for c in self.calls], ["q2"])  # q3 skipped: no tests
        self.assertEqual(len(pairs), 1)  # recovered from checkpoint + no new pairs
        self.assertEqual([e["idx"] for e in self._read_ckpt()], [0, 1, 2, 3])

    def test_malformed_line_is_reprocessed(self):
        with open(self.ckpt, "w", encoding="utf-8") as f:
            f.write(json.dumps({"idx": 0, "pair": {
                "prompt": "q0", "chosen": "code_AC_1", "rejected": "code_WA_0"}}) + "\n")
            f.write('{"idx": 1, "pair"')  # killed mid-write
        self.calls.clear()
        pairs = dpo_mod.make_preference_pairs(PROBLEMS, None, None,
                                              checkpoint_path=self.ckpt, max_gen_time=90)
        self.assertEqual([c["problem"] for c in self.calls], ["q1", "q2"])
        self.assertEqual(len(pairs), 1)
        self.assertEqual([e["idx"] for e in self._read_ckpt()], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
