"""Scientific invariants of the native C30 regional partition."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/preprocessing'))
import prepare_transcom_c30 as P


class PartitionTests(unittest.TestCase):
    def test_mixed_land_coast_and_unlabeled_cells(self):
        raw=np.zeros((11,4));raw[0,0]=3;raw[1,0]=1;raw[2,1]=.2;raw[10,3]=1
        w,missing=P.normalize_land_weights(raw,np.array([1,1,5,0]))
        np.testing.assert_allclose(w.sum(axis=0),1)
        np.testing.assert_allclose(w[:2,0],[.75,.25])
        self.assertEqual(w[2,1],1)  # no ocean fraction dropped at a coast
        self.assertEqual(w[5,2],1)  # closest labeled land is the fallback
        np.testing.assert_array_equal(missing,[False,False,True,False])

    def test_signed_fp32_flux_partition_closes(self):
        rng=np.random.default_rng(78)
        raw=rng.random((11,200));raw[:,3]=0
        w,_=P.normalize_land_weights(raw,np.zeros(200,dtype=int))
        flux=rng.normal(size=(17,200)).astype('f4')*1e-7
        regional=np.stack([(flux.astype('f8')*r).astype('f4') for r in w])
        np.testing.assert_allclose(regional.astype('f8').sum(axis=0),flux,rtol=2e-7,atol=0)
        self.assertTrue(np.all(regional[:,flux<0]<=0))
        self.assertTrue(np.all(regional[:,flux>0]>=0))

    def test_invalid_region_weights_rejected(self):
        raw=np.zeros((11,4));raw[0,0]=-1
        with self.assertRaises(AssertionError):P.normalize_land_weights(raw,np.zeros(4,dtype=int))


if __name__=='__main__':unittest.main()
