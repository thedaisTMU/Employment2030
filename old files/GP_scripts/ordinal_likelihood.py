# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews, Alexis Boukouvalas
# Copyright 2017 Artem Artemev @awav
# Copyright 2019 Jonathan Downing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

from gpflow.likelihoods import Likelihood
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter
from gpflow import transforms

def inv_probit(x,sigma=np.sqrt(2.0)):
    '''
    Inverse probit function.
    NB: do not take log of this function as it will result in underflow for large negative x.
    '''
    return 0.5 * (1.0 + tf.erf(x / sigma)) 

def log_inv_probit(x):
    '''
    This produces the correct asymptotic behaviour for log(inv_probit(x)) for large negative x
    '''

    ## All x's that are below a negative threshold are calculated using an asymptotic approximation.
    ## All x's above that threshold are calculated normally.

    # -1.45026 is the zero-crossing of the error between the approximation and truth, minimising the discontinuous jump when moving between the two.
    split_mask = tf.cast(tf.greater(x, -1.45026),tf.int32)
    split_idx = tf.dynamic_partition(tf.range(tf.shape(x)[0]),split_mask , 2)

    # Split the x values to the different piecewise regions
    x_split = tf.dynamic_partition(x,split_mask,2)
    
    # Approximation of log (inv_probit (x)) for x < -1.45026 
    y_0 = tf.log(1 - tf.exp(1.4 * x_split[0])) - (x_split[0]**2)/2  - tf.log(-x_split[0]) - np.log(1.136 * np.sqrt(2 * np.pi))

    # Use tf.log (inv_probit(x)) as it is accurate for x > -1.45026
    y_1 = tf.log(inv_probit(x_split[1]))
    
    # Stitch the results back together
    return tf.dynamic_stitch(split_idx,[y_0,y_1])

def log_difference_of_inv_probits(x,delta):    
    '''
    This produces the correct asymptotic behaviour for log(inv_probit(x+delta_half) - inv_probit(x-delta_half)) for both large negative and positive x.
    Also produces the correct asymptotic behaviour when delta -> 0.
    '''

    # Force to be strictly negative as the function is symmetric, assuming the observational noise for each label is equal.
    x = -tf.abs(x)
    
    delta_half = delta/2.
    
    ## When the value of delta is small a Taylor expansion approximation is used.
    ## Else function _log_difference_of_inv_probits_large_d is used.

    # Check where the values of delta_half are small.
    # The threshold in this case was set at 0.4, this was selected roughly to minimise error. Could be optimised.
    split_mask = tf.cast(tf.greater(delta_half, 0.4),tf.int32)
    split_idx = tf.dynamic_partition(tf.range(tf.shape(x)[0]),split_mask, 2)
    
    # Split the x and delta_half values to the different piecewise regions
    x_split = tf.dynamic_partition(x,split_mask,2)
    delta_half_split = tf.dynamic_partition(delta_half,split_mask,2)
    
    # In the small delta region use the Taylor expansion to 2nd order (x @ 0) of log(inv_probit(x+delta_half) - inv_probit(x-delta_half))
    low_d = -(delta_half_split[0] * tf.exp(-(delta_half_split[0]**2./2.))* x_split[0]**2.) \
                 /(np.sqrt(2. * np.pi) * tf.erf(delta_half_split[0]/np.sqrt(2.))) \
                 +tf.log(tf.erf(delta_half_split[0]/np.sqrt(2.)))
    
    # When delta is high
    high_d = _log_difference_of_inv_probits_large_d(x_split[1],delta_half_split[1])
    
    return tf.dynamic_stitch(split_idx,[low_d,high_d])
    
def _log_difference_of_inv_probits_large_d(x,delta_half):   
    '''
    For the case of when delta is large
    '''
    
    x_lower = x + delta_half
    x_upper = x - delta_half
    
    A = inv_probit(x_lower)
    B = inv_probit(x_upper)
    
    # Check where the value of x is outside the numerically stable region
    split_mask = tf.cast(tf.greater(A-B, 1e-6),tf.int32)
    split_idx = tf.dynamic_partition(tf.range(tf.shape(x)[0]),split_mask , 2)

    # Split the x and delta values to the different piecewise regions
    x_lower_split = tf.dynamic_partition(x_lower,split_mask,2)
    A_split = tf.dynamic_partition(A,split_mask,2)
    B_split = tf.dynamic_partition(B,split_mask,2)
    
    # Deals with cases when |x| >> 1 
    # Approximation of log (inv_probit (x)) from (An Improved Approximation for the Gaussian Q-Function - http://users.auth.gr/users/9/3/028239/public_html/pdf/Q_Approxim.pdf)
    y_0 = tf.log(1 - tf.exp(1.4 * x_lower_split[0])) - (x_lower_split[0]**2)/2  - tf.log(-x_lower_split[0]) - np.log(1.136 * np.sqrt(2 * np.pi))

    # Use tf.log (inv_probit(x) - inv_probit(y))) as it is accurate in this range
    y_1 = tf.log(A_split[1] - B_split[1])
    
    # Stitch the results back together
    return tf.dynamic_stitch(split_idx,[y_0,y_1])


def log_difference_of_inv_probits_complete(x,lower,upper):
    '''
    Takes in the input from GPflow, categorises them in to lowest, middle and highest label calling the appropriate function for each.
    '''

    x_shape = tf.shape(x)

    x = tf.reshape(x, [-1])
    lower = tf.reshape(lower, [-1])
    upper = tf.reshape(upper, [-1])
    
    ## Single Inv_probit for lowest labels
    # Check for lower boundary for -np.inf
    # 0 - Middle and highest labels
    # 1 - Lowest labels
    lower_infs = tf.cast(tf.equal(lower, -np.inf),tf.int32)

    lower_lowPart = tf.dynamic_partition(lower,lower_infs,2)
    upper_lowPart = tf.dynamic_partition(upper,lower_infs,2)
    x_lowPart = tf.dynamic_partition(x,lower_infs,2)
    
    lower_idx_range = tf.range(tf.shape(x)[0])
    lower_idx = tf.dynamic_partition(lower_idx_range,lower_infs , 2)

    # Calculate the log for the lowest labels
    y_lower_1 = log_inv_probit(upper_lowPart[1]-x_lowPart[1])

    ## Single Inv_probit for highest labels
    # Check for remaining upper boundary for np.inf
    # 0 - Middle labels
    # 1 - Highest labels
    upper_infs = tf.cast(tf.equal(upper_lowPart[0], np.inf),tf.int32)

    lower_upPart = tf.dynamic_partition(lower_lowPart[0],upper_infs,2)
    upper_upPart = tf.dynamic_partition(upper_lowPart[0],upper_infs,2)
    x_upPart = tf.dynamic_partition(x_lowPart[0],upper_infs,2)
    
    upper_idx_range = tf.range(tf.shape(x_lowPart[0])[0])
    upper_idx = tf.dynamic_partition(upper_idx_range,upper_infs , 2)

    # Calculate the log for the highest labels
    y_upper_1 = log_inv_probit(x_upPart[1] - lower_upPart[1])

    ## Double Inv_probit for middle labels
    # Shift x so it's symmetric around 0 and calculate the delta 
    delta_middle = upper_upPart[0] - lower_upPart[0]
    x_adjusted_middle = -x_upPart[0] + 0.5 * (upper_upPart[0] + lower_upPart[0])

    # Calculate the log for middle labels
    y_middle_0 = log_difference_of_inv_probits(x_adjusted_middle,delta_middle)

    ## Stitch everything back together!
    # Stitch middle and highest labels
    y_mid_up_0 = tf.dynamic_stitch(upper_idx,[y_middle_0,y_upper_1])
    # Stitch lowest labels with the rest
    y = tf.dynamic_stitch(lower_idx,[y_mid_up_0,y_lower_1])
    
    return tf.reshape(y,x_shape)

class Ordinal(Likelihood):
    """
    A likelihood for doing ordinal regression.

    The data are integer values from 0 to K, and the user must specify (K-1)
    'bin edges' which define the points at which the labels switch. Let the bin
    edges be [a_0, a_1, ... a_{K-1}], then the likelihood is

    p(Y=0|F) = phi((a_0 - F) / sigma)
    p(Y=1|F) = phi((a_1 - F) / sigma) - phi((a_0 - F) / sigma)
    p(Y=2|F) = phi((a_2 - F) / sigma) - phi((a_1 - F) / sigma)
    ...
    p(Y=K|F) = 1 - phi((a_{K-1} - F) / sigma)

    where phi is the cumulative density function of a Gaussian (the inverse inv_probit
    function) and sigma is a parameter to be learned. A reference is:

    @article{chu2005gaussian,
      title={Gaussian processes for ordinal regression},
      author={Chu, Wei and Ghahramani, Zoubin},
      journal={Journal of Machine Learning Research},
      volume={6},
      number={Jul},
      pages={1019--1041},
      year={2005}
    }

    JD: Aug 2019: Adapted more accurate log(inv_probit(x+delta_half) - inv_probit(x-delta_half)) 
    """

    def __init__(self, bin_edges, **kwargs):
        """
        bin_edges is a numpy array specifying at which function value the
        output label should switch. If the possible Y values are 0...K, then
        the size of bin_edges should be (K-1).
        """
        super().__init__(**kwargs)
        self.bin_edges = bin_edges
        self.num_bins = bin_edges.size + 1
        self.sigma = Parameter(1.0, transform=transforms.positive)

    @params_as_tensors
    def logp(self, F, Y):
        Y = tf.cast(Y, tf.int64)
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        
        lower = tf.gather(scaled_bins_right, Y)
        upper = tf.gather(scaled_bins_left, Y)

        # JD: This was changed from the original
        return log_difference_of_inv_probits_complete(F / self.sigma,lower,upper)

    @params_as_tensors
    def _make_phi(self, F):
        """
        A helper function for making predictions. Constructs a probability
        matrix where each row output the probability of the corresponding
        label, and the rows match the entries of F.

        Note that a matrix of F values is flattened.
        """
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        return inv_probit(scaled_bins_left - tf.reshape(F, (-1, 1)) / self.sigma) \
               - inv_probit(scaled_bins_right - tf.reshape(F, (-1, 1)) / self.sigma)

    def conditional_mean(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=np.float64), (-1, 1))
        return tf.reshape(tf.matmul(phi, Ys), tf.shape(F))

    def conditional_variance(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=np.float64), (-1, 1))
        E_y = tf.matmul(phi, Ys)
        E_y2 = tf.matmul(phi, tf.square(Ys))
        return tf.reshape(E_y2 - tf.square(E_y), tf.shape(F))