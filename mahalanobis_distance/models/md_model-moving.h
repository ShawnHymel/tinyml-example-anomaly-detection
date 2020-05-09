#ifndef MD_MODEL-MOVING_H
#define MD_MODEL-MOVING_H

const unsigned int model_mu_dim1 = 3;

const float model_mu[3] = {
    0.007695169087292815, 0.004149572143093924, 0.004909478363535906
};

const unsigned int model_inv_cov_dim1 = 3;
const unsigned int model_inv_cov_dim2 = 3;

const float model_inv_cov[3][3] = {
    94214.3873854578, -49201.81370745852, 2827.1091373699887, 
    -49201.813707458496, 152282.54724185396, -213180.62124189397, 
    2827.109137369931, -213180.6212418939, 804506.6877583836
};

#endif //MD_MODEL-MOVING_H