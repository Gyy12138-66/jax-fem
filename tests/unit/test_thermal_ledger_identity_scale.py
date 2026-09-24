"""The ledger's assembly-identity tolerance must scale with the largest term of the
identity, not with the (nearly cancelling) net total."""
import unittest

import numpy as np

from jax_fem_am.verification.thermal_ledger import assembly_identity_tolerance_j


class AssemblyIdentityToleranceTest(unittest.TestCase):
    def test_net_total_only_keeps_the_legacy_floor(self):
        # no term magnitudes: exactly the pre-2026-09-17 rule
        self.assertAlmostEqual(assembly_identity_tolerance_j(0.009, 0.009), 1.0e-12 + 1.0e-10 * 0.009, places=20)
        self.assertAlmostEqual(assembly_identity_tolerance_j(0.0, 0.0), 1.0e-12 + 1.0e-10 * np.finfo(np.float64).tiny, places=20)

    def test_large_cancelling_terms_raise_the_tolerance(self):
        # 0911 flash sub-step: storage ~ laser ~ 6.7e4 J, net ~ 1e-2 J, identity roundoff 3.9e-12 J
        tol = assembly_identity_tolerance_j(0.009, 0.009, term_magnitudes_j=(66769.18, -66769.18, 0.0, 0.0, 0.0))
        self.assertGreater(tol, 3.9e-12)
        self.assertAlmostEqual(tol, 1.0e-12 + 1.0e-10 * 66769.18, places=15)
        # still six orders above the double-precision roundoff of the largest term
        self.assertLess(64 * np.finfo(np.float64).eps * 66769.18, tol)

    def test_0119_flash_step_unchanged_in_outcome(self):
        # 0119: 1,220 J flash, identity error 1.3e-13 J passed before and passes now
        tol = assembly_identity_tolerance_j(0.001, 0.001, term_magnitudes_j=(1220.6, -1220.6))
        self.assertLess(1.3e-13, tol)

    def test_real_identity_violations_are_still_caught(self):
        # an error of 1e-6 J on a 100 J step is 1e-8 relative: far above 1e-10, must fail
        tol = assembly_identity_tolerance_j(0.5, 0.5, term_magnitudes_j=(100.0, -99.5))
        self.assertLess(tol, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
