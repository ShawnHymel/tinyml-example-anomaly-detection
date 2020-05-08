#ifndef MD_UTILS_H
#define MD_UTILS_H

int compare_floats(const void *p, const void *q);
float median(float *arr, int arr_len);
float calc_mad(float *arr, int arr_len);
float dot_product(float *a, float *b, int a_len, int b_len);
int matrix_multiply(const float *a, 
                    const float *b, 
                    int a_rows, 
                    int a_cols, 
                    int b_rows, 
                    int b_cols,
                    float *prod);
float mahalanobis(const float *x, 
                  const float *mu, 
                  const float *inv_cov, 
                  int len);
float calc_mse(const float *x, const float *x_hat, const int len);

#endif //MD_UTILS_H
