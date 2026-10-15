"""Scientific invariants: relative coordinates, causal timing and grouped folds."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from contactworld_slip_core import grasp, labels, targets, causal_baseline, action_schedule, KINDS
from probe_contactworld_slip import components, splits
from branch_contactworld_slip import match


class SlipTests(unittest.TestCase):
    def test_rigid_grasp_under_large_arm_rotation(self):
        q=Rotation.from_euler('xyz',np.array([[0,0,0],[.3,.2,-.5],[1.,-.3,.2]]))
        ee=np.array([[0.,0.,0.],[.1,.2,.3],[.3,.1,.7]])
        plug=ee+q.apply(np.tile([.02,-.01,.06],(3,1)))
        pq=(q*Rotation.from_euler('y',.5)).as_quat()
        p,r=grasp(ee,q.as_quat(),plug,pq)
        np.testing.assert_allclose(p,np.tile([20,-10,60],(3,1)),atol=1e-10)
        self.assertFalse(labels(p,r)['active'].any())

    def test_labels_are_causal_and_forecast_is_not_detection(self):
        p=np.zeros((20,3));p[9:,0]=np.arange(11)*.7
        q=np.tile([0,0,0,1.],(20,1));a,y,v=targets(p,q)
        for stop in range(3,20):
            np.testing.assert_array_equal(labels(p[:stop],q[:stop])['active'],a['active'][:stop])
        self.assertTrue(y['forecast4'][8]);self.assertFalse(a['active'][8])
        self.assertFalse(v['forecast4'][a['active']].any())
        self.assertFalse(v['forecast4'][-4:].any())
        self.assertFalse(v['quiet4'][a['movement']].any())

    def test_inputs_cannot_see_future_action_or_baseline(self):
        f=np.ones((12,840),np.float32)*.001;s=np.zeros((12,18),np.float32);a=np.zeros((11,6),np.float32)
        b,v=causal_baseline(f,s,a);c=components(f,s,a,b,v,np.array([5]))
        a[5:]=.9;f[6:]*=20
        b2,v2=causal_baseline(f,s,a);c2=components(f,s,a,b2,v2,np.array([5]))
        for k in c:np.testing.assert_allclose(c[k],c2[k])
        self.assertTrue(v[5]);self.assertFalse(v[:4].any())

    def test_group_folds_keep_all_models_of_seed_together(self):
        groups=np.repeat(np.arange(40),13)
        covered=np.zeros(len(groups),int)
        for _,tr,va,te in splits(groups):
            self.assertFalse(set(groups[tr])&set(groups[va]));self.assertFalse(set(groups[tr])&set(groups[te]))
            self.assertFalse(set(groups[va])&set(groups[te]));covered+=te
        np.testing.assert_array_equal(covered,np.ones(len(groups)))

    def test_actions_resume_same_known_commands(self):
        plan=np.arange(24).reshape(4,6).astype(float)/100
        continued=action_schedule('continue',plan,[0,0,0,1],np.ones(3),np.ones(3))
        np.testing.assert_allclose(continued,np.tile(plan,(4,1)))
        pause=action_schedule('pause_resume',plan,[0,0,0,1],np.ones(3),np.ones(3))
        np.testing.assert_allclose(pause[:4],0);np.testing.assert_allclose(pause[4:],continued[4:])
        for kind in KINDS:self.assertEqual(action_schedule(kind,plan,[0,0,0,1],np.ones(3),np.ones(3)).shape,(16,6))

    def test_reconstruction_gate_rejects_small_unmatched_motion(self):
        a={k:np.zeros(3) for k in ('plug_pos','ee_pos','socket_pos_gt')}
        a.update({k:np.array([0,0,0,1.]) for k in ('plug_quat','ee_quat','socket_quat')})
        a.update(dof_pos=np.zeros(9),dof_vel=np.zeros(9))
        b={k:v.copy() for k,v in a.items()};f=np.ones((2,10,14,3))*.001
        self.assertTrue(match(a,b,f,f)['pass_gate']);b['plug_pos'][2]+=.0003
        self.assertFalse(match(a,b,f,f)['pass_gate'])


if __name__=='__main__':unittest.main()
