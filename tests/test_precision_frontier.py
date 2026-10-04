import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from task01_precision_frontier import largest_remainder, choose, counts, training_view
import moldguard as mg


class PrecisionContractTests(unittest.TestCase):
    def test_exact_allocation_and_label_free_ties(self):
        self.assertEqual(sum(largest_remainder({0:193,1:194,2:194,3:193,4:193},73).values()),73)
        frame=pd.DataFrame({'machine':['rg3']*20,'source_row_id':range(20),'batch':np.repeat([0,1],10)})
        d=choose(frame,np.ones(20),.15)
        self.assertEqual(np.flatnonzero(d).tolist(),[0,1,10])
        order=np.random.default_rng(0).permutation(20)
        shuffled=choose(frame.iloc[order],np.ones(20),.15)
        self.assertEqual(sorted(frame.iloc[order].loc[shuffled,'source_row_id']),[0,1,10])

    def test_binomial_bce_and_gradient(self):
        n=np.array([2,2,3]);c=np.array([1,0,3]);z=np.array([-.8,.4,1.3]);p=expit(z)
        y=np.concatenate([np.r_[np.ones(a),np.zeros(b-a)] for a,b in zip(c,n)])
        repeated=np.repeat(p,n)
        row=-np.sum(y*np.log(repeated)+(1-y)*np.log1p(-repeated))
        soft=-np.sum(n*((c/n)*np.log(p)+(1-c/n)*np.log1p(-p)))
        self.assertAlmostEqual(row,soft,places=12)
        np.testing.assert_allclose(np.add.reduceat(repeated-y,np.r_[0,np.cumsum(n)[:-1]]),n*p-c,atol=1e-12)

    def test_event_target_is_not_row_target(self):
        f=pd.DataFrame({'feature_group':['g','g','h','h'],mg.TARGET:[0,1,0,0]})
        self.assertEqual(training_view(f,'row')[mg.TARGET].tolist(),[0,1,0,0])
        self.assertEqual(training_view(f,'event')[mg.TARGET].tolist(),[1,0])

    def test_fixed_k_metric_identities(self):
        result=counts([1,0,1,0,0],[1,1,0,0,0])
        self.assertEqual([result[x] for x in ['tp','fp','fn','tn']],[1,1,1,2])
        self.assertAlmostEqual(result['accuracy'],.6)
        self.assertAlmostEqual(result['fdr'],.5)
        self.assertAlmostEqual(result['fpr'],1/3)

    def test_nonfinite_score_rejected(self):
        f=pd.DataFrame({'machine':['rg3'],'source_row_id':[1],'batch':[0]})
        with self.assertRaises(ValueError): choose(f,[np.nan])


if __name__=='__main__': unittest.main()
