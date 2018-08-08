import numpy as np
import env_utils as envu

class Reward(object):

    """
        Minimizes Velocity Field Tracking Error

    """

    def __init__(self, reward_scale=1.0, ts_coeff=-0.0,  fuel_coeff=-0.05,
                 landing_rlimit=5.0, landing_vlimit=2.0,
                 tracking_coeff=-0.01, tracking_bias=0.01):

        self.reward_scale =         reward_scale
        self.ts_coeff =             ts_coeff
        self.fuel_coeff =           fuel_coeff

        self.landing_rlimit =       landing_rlimit
        self.landing_vlimit =       landing_vlimit

        self.tracking_coeff =       tracking_coeff
        self.tracking_bias =        tracking_bias

        self.quad_reward =          self.quad_reward1

    def get(self, lander,  action, done, steps, shape_constraint, glideslope_constraint, attitude_constraint):
        pos         =  lander.state['position']
        vel         =  lander.state['velocity']

        prev_pos    =  lander.prev_state['position']
        prev_vel    =  lander.prev_state['velocity']

        state = np.hstack((pos,vel))
        prev_state = np.hstack((prev_pos,prev_vel))

        r_gs = glideslope_constraint.get_reward()

        r_sc, sc_margin = shape_constraint.get_reward(lander.state)

        tracking_error, t_go = lander.track_func(pos, vel)
        r_tracking  =  self.tracking_bias + self.tracking_coeff * np.linalg.norm(tracking_error)

        r_att = attitude_constraint.get_reward(lander.state)

        landing_margin = 0.
        gs_penalty = 0.0
        sc_penalty = 0.0
        att_penalty = 0.0
        if done:
            gs_penalty = glideslope_constraint.get_term_reward()

            att_penalty = attitude_constraint.get_term_reward(lander.state)

            sc_penalty = shape_constraint.get_term_reward(lander.state)

            landing_margin = np.maximum(np.linalg.norm(pos) -  self.landing_rlimit , np.linalg.norm(vel) -  self.landing_vlimit)

        reward_info = {}

        r_fuel = self.fuel_coeff * np.sum(lander.state['bf_thrust']) / (lander.thruster_model.num_thrusters*lander.thruster_model.max_thrust)

        reward_info['fuel'] = r_fuel

        reward = (r_att + att_penalty + sc_penalty + gs_penalty + r_gs + r_sc +  r_tracking +  r_fuel + self.ts_coeff) * self.reward_scale
        lander.trajectory['reward'].append(reward)

        lander.trajectory['glideslope'].append(glideslope_constraint.get())
        lander.trajectory['glideslope_reward'].append(r_gs)
        lander.trajectory['glideslope_penalty'].append(gs_penalty)
        lander.trajectory['att_reward'].append(r_att)
        lander.trajectory['att_penalty'].append(att_penalty)
        lander.trajectory['sc_penalty'].append(sc_penalty)
        lander.trajectory['sc_margin'].append(sc_margin)
        lander.trajectory['sc_reward'].append(r_sc)
        lander.trajectory['tracking_reward'].append(r_tracking)
        lander.trajectory['landing_margin'].append(landing_margin)
        lander.trajectory['fuel_reward'].append(r_fuel)
        return reward, reward_info

    def quad_reward1(self,val,sigma):
        val = np.linalg.norm(val)
        reward = 1 + np.maximum( -1, -(val/sigma)**2)
        return reward

    def quad_reward2(self,val,sigma):
        val = np.linalg.norm(val)
        reward =  -(val/sigma)**2
        return reward

    def quad_reward3(self,val,sigma):
        val = np.linalg.norm(val)
        reward = 1 - (val/sigma)**2
        return reward

