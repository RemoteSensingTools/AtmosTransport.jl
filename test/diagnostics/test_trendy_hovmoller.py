"""Scientific invariants for the TRENDY seasonal/IAV comparison.

Run with python3 -m unittest discover -s test/diagnostics -p test_trendy_hovmoller.py
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/diagnostics/rank_trendy_hovmoller.py'
spec = importlib.util.spec_from_file_location('rankhov', SCRIPT)
rankhov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rankhov)


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(47)
        t = (np.arange(120)-59.5)/12
        cycle = 3*np.sin(np.arange(120)*2*np.pi/12)
        self.x = (400+.7*t+cycle)[:, None]*np.ones((1,40))
        self.seen = self.rng.random((120,40))>.15
        self.seen[:,0] = np.arange(120)%12<7

    def test_trend_and_season_removed_with_polar_gaps(self):
        anomaly, seasonal, _ = rankhov.decompose(self.x[None], self.seen)
        self.assertLess(np.nanmax(np.abs(anomaly)),1e-10)
        self.assertTrue(np.isnan(seasonal[:,7:,0]).all())
        self.assertTrue(np.isnan(anomaly[0][~self.seen]).all())

    def test_carriers_do_not_change_anomalies(self):
        noisy = self.x+self.rng.normal(size=self.x.shape)
        a,s,_ = rankhov.decompose(np.stack([noisy,noisy+1300]),self.seen)
        np.testing.assert_allclose(a[0],a[1],atol=1e-10)
        np.testing.assert_allclose(s[0],s[1],atol=1e-10)

    def test_fire_addition_commutes_with_shared_projection(self):
        fire = self.rng.normal(size=self.x.shape)
        a,s,d = rankhov.decompose(np.stack([self.x,fire,self.x+fire]),self.seen)
        for fields in (a,s,d):
            np.testing.assert_allclose(fields[0]+fields[1],fields[2],atol=1e-10)

    def test_correlation_and_rmse_distinguish_pattern_from_amplitude(self):
        x = np.arange(20,dtype=float)-9.5
        doubled = rankhov.metrics(2*x,x)
        self.assertAlmostEqual(doubled['r'],1)
        self.assertAlmostEqual(doubled['amplitude_ratio'],2)
        self.assertGreater(doubled['rmse_ppm'],0)
        self.assertAlmostEqual(rankhov.metrics(-x,x)['r'],-1)
        self.assertEqual(rankhov.metrics(x,x)['rmse_ppm'],0)


if __name__ == '__main__':
    unittest.main()
