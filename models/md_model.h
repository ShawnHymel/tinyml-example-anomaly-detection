#ifndef MD_MODEL_H
#define MD_MODEL_H

const unsigned int model_mu_dim1 = 3;

const float model_mu[3] = {
    0.012049953994017095, 0.00709583131025641, 0.004973618240170936
};

const unsigned int model_inv_cov_dim1 = 3;
const unsigned int model_inv_cov_dim2 = 3;

const float model_inv_cov[3][3] = {
    1927196.2578237287, -324748.88969416154, -199864.16316800224, 
    -324748.88969416154, 2157143.906105444, -40965.82067724897, 
    -199864.16316800224, -40965.82067724897, 3192311.649781269
};

#endif //MD_MODEL_H