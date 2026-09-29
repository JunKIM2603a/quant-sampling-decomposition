import math
import unittest
from fractions import Fraction
import numpy as np
from quantsplit.data import make_splits, manifest_hash, normalize_question
from quantsplit.evaluation import score_gsm8k, numeric_value
from quantsplit.generation import Settings, generate_four, generate_arm
from quantsplit.sampling import (ARMS, PolicyPair, inverse_cdf, keyed_uniform,
                                 nucleus_mask, probabilities, restrict)


class RecordingBackend:
    def __init__(self, offset=0, force_eos=False):
        self.offset, self.force_eos, self.sessions = offset, force_eos, []

    def new_session(self):
        backend = self
        class State:
            def __init__(self):
                self.prefix = ()
                self.seen = []
            def next_logits(self, prefix):
                prefix = tuple(prefix)
                if prefix[:len(self.prefix)] != self.prefix or len(prefix) <= len(self.prefix):
                    raise ValueError("cache contaminated or double-fed")
                self.prefix = prefix
                self.seen.append(prefix)
                if backend.force_eos:
                    return np.array([-100, -100, 100, -100], dtype=np.float32)
                # Full prefix checksum makes accidental foreign histories observable.
                x = sum((i+1)*t for i,t in enumerate(prefix)) + backend.offset
                return np.array([math.sin(x+i*1.7)*2 for i in range(7)], dtype=np.float32)
        state = State()
        self.sessions.append(state)
        return state


class SamplingTests(unittest.TestCase):
    def test_crossed_policy_matches_hand_computed_example(self):
        pf = np.array([.5,.3,.15,.05], dtype=np.float32)
        pq = np.array([.1,.2,.3,.4], dtype=np.float32)
        pair = PolicyPair(pf, pq, nucleus_mask(pf,.7), nucleus_mask(pq,.7))
        np.testing.assert_array_equal(pair.mf,[1,1,0,0])
        np.testing.assert_array_equal(pair.mq,[0,0,1,1])
        np.testing.assert_allclose(pair.distribution("QF"),[1/3,2/3,0,0],atol=1e-7)
        np.testing.assert_allclose(pair.distribution("FQ"),[0,0,.75,.25],atol=1e-7)

    def test_boundary_and_ties(self):
        p=np.array([.4,.3,.3],dtype=np.float32)
        np.testing.assert_array_equal(nucleus_mask(p,.7),[1,1,0])
        np.testing.assert_array_equal(nucleus_mask(np.full(4,.25),.5),[1,1,0,0])

    def test_top_p_one_includes_zero_probability_tokens(self):
        np.testing.assert_array_equal(nucleus_mask([1,0,0],1),[1,1,1])

    def test_same_logits_four_distributions_identical(self):
        pair=PolicyPair.from_logits([3,2,-3,1],[3,2,-3,1])
        for arm in ARMS:
            np.testing.assert_array_equal(pair.distribution(arm),pair.distribution("FF"))

    def test_no_second_top_p(self):
        pair=PolicyPair.from_logits([3,2,1,0],[0,1,2,3],temperature=1,top_p=.8)
        self.assertEqual(np.count_nonzero(pair.distribution("QF")),int(pair.mf.sum()))
        self.assertAlmostEqual(pair.distribution("QF")[0]/pair.distribution("QF")[1],math.exp(-1),places=6)

    def test_errors_not_zero_length_results(self):
        for bad in [[np.nan,0],[np.inf,0],[]]:
            with self.assertRaises(ValueError): probabilities(bad)
        with self.assertRaises(ValueError): restrict([1,0],[0,1])
        with self.assertRaises(ValueError): nucleus_mask([.5,.5],0)
        with self.assertRaises(ValueError): PolicyPair.from_logits([1],[1,2])

    def test_inverse_cdf_skips_zero_mass_and_last_rounding(self):
        self.assertEqual(inverse_cdf([0,.5,.5,0],0),1)
        self.assertEqual(inverse_cdf([0,.5,.5,0],.5),2)
        self.assertEqual(inverse_cdf([0,.5,.5,0],np.nextafter(1.,0.)),2)

    def test_tv_identity_on_independent_random_vectors(self):
        rng=np.random.default_rng(8)
        for _ in range(30):
            pair=PolicyPair.from_logits(rng.normal(size=31),rng.normal(size=31),top_p=.8)
            d=pair.diagnostics()
            self.assertAlmostEqual(d["tv_qq_qf"],d["tv_identity"],places=6)

    def test_rng_order_seed_and_step(self):
        before={(q,s):keyed_uniform(q,42,s) for q in ["a","b"] for s in range(10)}
        after={(q,s):keyed_uniform(q,42,s) for q in ["b","a"] for s in reversed(range(10))}
        self.assertEqual(before,after)
        self.assertNotEqual(keyed_uniform("a",42,0),keyed_uniform("a",43,0))
        self.assertTrue(all(0<=v<1 for v in before.values()))

    def test_sampler_marginals_on_fixed_uniform_grid(self):
        # Deterministic 10000-point integration; does not depend on a lucky RNG draw.
        p=[.1,.2,.3,.4]
        counts=np.bincount([inverse_cdf(p,(i+.5)/10000) for i in range(10000)],minlength=4)
        np.testing.assert_array_equal(counts,[1000,2000,3000,4000])


class GenerationTests(unittest.TestCase):
    def test_each_arm_owns_two_caches_same_current_prefix(self):
        f,q=RecordingBackend(),RecordingBackend(offset=7)
        results=generate_four(f,q,[1,2],"toy",42,Settings(max_new_tokens=8))
        self.assertEqual(len(f.sessions),4)
        self.assertEqual(len({id(s) for s in f.sessions+q.sessions}),8)
        for i,arm in enumerate(ARMS):
            self.assertEqual(f.sessions[i].seen,q.sessions[i].seen)
            self.assertEqual(f.sessions[i].seen[-1],tuple([1,2]+results[arm].generated_ids[:-1]))
        self.assertGreater(len({tuple(r.generated_ids) for r in results.values()}),1)

    def test_equal_backends_and_top_p_one_rollout_identity(self):
        f=RecordingBackend()
        same=generate_four(f,f,[1,2],"toy",42,Settings(max_new_tokens=20))
        self.assertEqual(len({tuple(v.generated_ids) for v in same.values()}),1)
        other=generate_four(f,RecordingBackend(5),[1,2],"toy",42,Settings(max_new_tokens=20,top_p=1))
        self.assertEqual(other['FF'].generated_ids,other['FQ'].generated_ids)
        self.assertEqual(other['QQ'].generated_ids,other['QF'].generated_ids)

    def test_arm_order_does_not_change_outputs(self):
        f,q=RecordingBackend(),RecordingBackend(4)
        a=generate_four(f,q,[1,2],"toy",42,Settings(max_new_tokens=8))
        b=generate_four(f,q,[1,2],"toy",42,Settings(max_new_tokens=8),tuple(reversed(ARMS)))
        self.assertEqual({k:v.generated_ids for k,v in a.items()},{k:v.generated_ids for k,v in b.items()})

    def test_eos_included_and_reasoning_end_not_stop(self):
        f=RecordingBackend(force_eos=True)
        eos=generate_arm(f,f,[1],"toy",42,"FF",Settings(max_new_tokens=1,eos_ids=(2,)))
        self.assertEqual((eos.consumed_tokens,eos.stop_reason),(1,"eos"))
        cap=generate_arm(f,f,[1],"toy",42,"FF",Settings(max_new_tokens=3,diagnostic_ids=(2,)))
        self.assertEqual((cap.consumed_tokens,cap.stop_reason),(3,"cap"))


class ParserTests(unittest.TestCase):
    def test_reasoning_number_never_counts(self):
        self.assertFalse(score_gsm8k(r"<think>\boxed{3}","#### 3").correct)
        self.assertFalse(score_gsm8k(r"\boxed{3}</think>no answer","#### 3").correct)

    def test_final_box_after_last_reasoning_end(self):
        self.assertTrue(score_gsm8k(r"\boxed{7}</think>\boxed{4} or \boxed{3}","#### 3").correct)
        self.assertFalse(score_gsm8k(r"</think>\boxed{3}\boxed{4", "#### 3").correct)

    def test_numeric_formats_and_rejection(self):
        for raw in [r"\frac{1}{2}",r"\dfrac{1}{2}",".5","1/2","0.50"]:
            self.assertEqual(numeric_value(raw),Fraction(1,2))
        self.assertEqual(numeric_value("1,200"),1200)
        for raw in ["12,00","3 meters","50%","1+2",r"\frac{1}{0}"]:
            with self.assertRaises((ValueError,ZeroDivisionError)): numeric_value(raw)

    def test_complete_answer_scores_even_when_capped(self):
        # Stop reason is intentionally independent of fixed-budget correctness.
        self.assertTrue(score_gsm8k(r"</think>\boxed{3}","work #### 3").correct)
        with self.assertRaises(ValueError): score_gsm8k(r"</think>\boxed{3}","bad dataset label")


class DataTests(unittest.TestCase):
    def test_disjoint_splits_deterministic_exact_near_audit(self):
        train=[{"question":f"Unique topic {i} owns alpha{i} beta{i} gamma{i} delta{i}"} for i in range(12)]
        train += [{"question":"Ａ  QUESTION"},{"question":"a question"},
                  {"question":"one two three four five six seven eight nine ten eleven twelve extra"}]
        test=[{"question":"a question"},{"question":"one two three four five six seven eight nine ten eleven twelve"}]
        a=make_splits(train,test,n=3)
        b=make_splits(train,test,n=3)
        self.assertEqual(manifest_hash(a),manifest_hash(b))
        ids=[i["row_index"] for split in ['calibration','development','pilot'] for i in a[split]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(a['excluded']),3)
        self.assertEqual(normalize_question('Ａ  QUESTION'),'a question')


if __name__ == '__main__':
    unittest.main()
