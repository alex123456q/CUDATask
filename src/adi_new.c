#include "header.h"

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                  
//---------------------------------------------------------------------
/*#define us(y, z, m) us[m + (z)*P_SIZE + (y)*P_SIZE*P_SIZE]
#define vs(y, z, m) vs[m + (z)*P_SIZE + (y)*P_SIZE*P_SIZE]
#define ws(y, z, m) ws[m + (z)*P_SIZE + (y)*P_SIZE*P_SIZE]
#define qs(y, z, m) qs[m + (z)*P_SIZE + (y)*P_SIZE*P_SIZE]
#define rho_i(y, z, m) rho_i[m + (z)*P_SIZE + (y)*P_SIZE*P_SIZE]
#define speed(y, z, m) speed[m + (z)*P_SIZE + (y)*P_SIZE*P_SIZE]
#define rhs(x, y, z, m) rhs[m + (z)*5 + (y)*5*P_SIZE + (x)*5*P_SIZE*P_SIZE]
__global__
void xinvr_kernel(double* us, double* vs, double* ws, double *qs, double* rho_i, double *speed, double *rhs, const int nx2, const int ny2, const int nz2, const double bt, const double c2)
{
    int i, j, k;
    i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

    if (i <= nx2 && j <= ny2 && k <= nz2){
        ru1 = rho_i(k, j, i);
        uu = us(k, j, i);
        vv = vs(k, j, i);
        ww = ws(k, j, i);
        ac = speed(k, j, i);
        ac2inv = ac*ac;

        r1 = rhs(k, j, i, 0);
        r2 = rhs(k, j, i, 1);
        r3 = rhs(k, j, i, 2);
        r4 = rhs(k, j, i, 3);
        r5 = rhs(k, j, i, 4);

        t1 = c2 / ac2inv * (qs(k, j, i) * r1 - uu*r2 - vv*r3 - ww*r4 + r5);
        t2 = bt * ru1 * (uu * r1 - r2);
        t3 = (bt * ru1 * ac) * t1;

        rhs(k, j, i, 0) = r1 - t1;
        rhs(k, j, i, 1) = -ru1 * (ww*r1 - r4);
        rhs(k, j, i, 2) = ru1 * (vv*r1 - r3);
        rhs(k, j, i, 3) = -t2 + t3;
        rhs(k, j, i, 4) = t2 + t3;
    }
}
#undef us
#undef vs
#undef ws
#undef qs
#undef speed
#undef rho_i
#undef rhs*/


void xinvr()
{
    int i, j, k;
    double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

    if (timeron) timer_start(t_txinvr);

   for (k = 1; k <= nz2; k++)
    {
        for (j = 1; j <= ny2; j++)
        {            
            for (i = 1; i <= nx2; i++)
            {
                ru1 = rho_i[k][j][i];
                uu = us[k][j][i];
                vv = vs[k][j][i];
                ww = ws[k][j][i];
                ac = speed[k][j][i];
                ac2inv = ac*ac;

                r1 = rhs[k][j][i][0];
                r2 = rhs[k][j][i][1];
                r3 = rhs[k][j][i][2];
                r4 = rhs[k][j][i][3];
                r5 = rhs[k][j][i][4];

                t1 = c2 / ac2inv * (qs[k][j][i] * r1 - uu*r2 - vv*r3 - ww*r4 + r5);
                t2 = bt * ru1 * (uu * r1 - r2);
                t3 = (bt * ru1 * ac) * t1;

                rhs[k][j][i][0] = r1 - t1;
                rhs[k][j][i][1] = -ru1 * (ww*r1 - r4);
                rhs[k][j][i][2] = ru1 * (vv*r1 - r3);
                rhs[k][j][i][3] = -t2 + t3;
                rhs[k][j][i][4] = t2 + t3;
            }
        }
    }


/*    const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE;
    dim3 blocks = dim3(nx2/32+1, ny2/4+1, nz2);
    dim3 threads = dim3(32, 4, 1);
    SAFE_CALL(cudaMemcpy(gpuUs, us, size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpuVs, vs, size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpuWs, ws, size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpuQs, qs, size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpuRho_i, rho_i, size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpuSpeed, speed, size, cudaMemcpyHostToDevice));


    SAFE_CALL(cudaMemcpy(gpuRhs, rhs, size*5, cudaMemcpyHostToDevice));

    xinvr_kernel<<<blocks, threads>>>(gpuUs, gpuVs, gpuWs, gpuQs, gpuRho_i, gpuSpeed,gpuRhs, nx2, ny2, nz2, bt, c2);
    //SAFE_CALL

    SAFE_CALL(cudaMemcpy(rhs, gpuRhs, size*5, cudaMemcpyDeviceToHost));*/
    if (timeron) timer_stop(t_txinvr);
}


#define u(x, y, z, m) u[m + (z)*5 + (y)*5*P_SIZE + (x)*5*P_SIZE*P_SIZE]
#define rhs(x, y, z, m) rhs[m + (z)*5 + (y)*5*P_SIZE + (x)*5*P_SIZE*P_SIZE]
__global__
void add_kernel(double* u, double *rhs, const int nx2, const int ny2, const int nz2)
{
    int i, j, k, m;
    i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2){
        for (m = 0; m < 5; m++) {
            u(k, j, i, m) = u(k, j, i, m) + rhs(k, j, i, m);
        }
    }
}
#undef u
#undef rhs

void add()
{
    if (timeron) timer_start(t_add);

//    for (int i = 0; i < count; ++i)
//    {
//        for (int i = 0; i < count; ++i)
//        {
//            for (int i = 0; i < count; ++i)
//            {
//                for (int i = 0; i < count; ++i)
//                {
//                    u_tmp[m][k][j][i] = u[k][j][i][m]
//                }
//            }
//        }
//    }

    /*int i, j, k, m;
    for (k = 1; k <= nz2; k++) {
        for (j = 1; j <= ny2; j++) {
            for (i = 1; i <= nx2; i++) {
                for (m = 0; m < 5; m++) {
                    u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
                }
            }
        }
    }*/
    const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE*5;
    dim3 blocks = dim3(nx2/32+1, ny2/4+1, nz2);
    dim3 threads = dim3(32, 4, 1);
    SAFE_CALL(cudaMemcpy(gpuU, u, size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpuRhs, rhs, size, cudaMemcpyHostToDevice));

    add_kernel<<<blocks, threads>>>(gpuU, gpuRhs, nx2, ny2, nz2);
    //SAFE_CALL

    SAFE_CALL(cudaMemcpy(u, gpuU, size, cudaMemcpyDeviceToHost));
    if (timeron) timer_stop(t_add);
}

void adi()
{
    compute_rhs();
    xinvr();
    x_solve();
    y_solve();
    z_solve();
    add();
}
