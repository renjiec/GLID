#include "utils.cuh"
#include <math_constants.h>

//#include <cub/block/block_load.cuh>
//#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

enum LINEAR_SOLVE_PREFERENCE { LINEAR_SOLVE_PREFER_CHOLESKY = 0, LINEAR_SOLVE_PREFER_LU = 1, LINEAR_SOLVE_FORCE_CHOLESKY = 2 };

const int threadsPerBlock = 256;
const int nItemPerReduceThread = 10;

const __device__ double minDeformStepNorm = 1e-12;

inline __host__ __device__ int blockNum(int n) { return ceildiv(n, threadsPerBlock); }   // used in line search
inline __host__ __device__ int reduceBlockNum(int n) { return ceildiv(n, threadsPerBlock*nItemPerReduceThread); }

//////////////////////////////////////////////////////////////////////////
// take sqrt of each entry, if negative, take 0, utilily for modify a symmetric matrix (hessian) to be SPD
template<class R>
__global__ void sqrt_non_negative_clamp(const R *x, R *y, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= n) return;
    y[i] = x[i] > 0 ? sqrt(x[i]) : 0;
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void abs_diff_similarity_polygon(real *dSimlarity, const Complex *phi, const Complex *dphi, const Complex *v, int n, const real *t)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;


    //if (i >= n*2) return;
    //const int next = (i + 1) % n + n - (i < n)*n;
    //const int prev = (i + n - 1) % n + n - (i < n)*n;

    //Complex s1 = (phi[next] - phi[i]) / (v[next] - v[i]);
    //Complex s2 = (phi[prev] - phi[i]) / (v[prev] - v[i]);
    //dSimlarity[i] = abs(s1 - s2);

    //s1 += (dphi[next] - dphi[i])**t / (v[next] - v[i]);
    //s2 += (dphi[prev] - dphi[i])**t / (v[prev] - v[i]);
    //dSimlarity[i+n*2] = abs(s1 - s2);


    if (i >= n) return;
    const int next = (i < n - 1) ? i + 1 : 0;
    const int prev = (i > 0) ? i - 1 : n - 1;

    Complex s1 = (phi[next] - phi[i]) / (v[next] - v[i]);
    Complex s2 = (phi[prev] - phi[i]) / (v[prev] - v[i]);
    dSimlarity[i] = abs(s1 - s2);

    s1 += (dphi[next] - dphi[i])**t / (v[next] - v[i]);
    s2 += (dphi[prev] - dphi[i])**t / (v[prev] - v[i]);
    dSimlarity[i+n*2] = abs(s1 - s2);


    const Complex* psy = phi + n;
    const Complex* dpsy = dphi + n;

    s1 = (psy[next] - psy[i]) / (v[next] - v[i]);
    s2 = (psy[prev] - psy[i]) / (v[prev] - v[i]);
    dSimlarity[i+n] = abs(s1 - s2);

    s1 += (dpsy[next] - dpsy[i])**t / (v[next] - v[i]);
    s2 += (dpsy[prev] - dpsy[i])**t / (v[prev] - v[i]);
    dSimlarity[i+n*3] = abs(s1 - s2);
}

//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void p2p_energy(real *en, const Complex *fg, const Complex *dfg, const real *t, const Complex *b, int n, real lambda)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0;

    const int ien = blockIdx.y;
    if (i < n) {
        if (dfg) {
            e += abs2( fg[i]+dfg[i]*t[ien]+conj(fg[i+n]+dfg[i+n]*t[ien])-b[i] );
        }
        else {
            e += abs2( fg[i]+conj(fg[i+n])-b[i] );
        }
    }

    using BlockReduceT = cub::BlockReduce<double, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    real block_en = real( lambda*BlockReduceT(temp_storage).Sum(e) );

    if (threadIdx.x == 0)
        atomicAdd(en+ien, block_en);
}



//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void isometry_energy(real *en, const Complex *fzgz, const Complex *dfzgz, const real *t, int m, real epow)
{
	const int it = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0;

    const int ien = blockIdx.y;
#pragma unroll
    for (int j = 0; j < nItemPerReduceThread; j++) {
        int i = it*nItemPerReduceThread + j;
        if (i >= m) continue;
        double ee = 0;
        if (dfzgz) {
            double fz2 = abs2(fzgz[i] + dfzgz[i]*t[ien]);
            double gz2 = abs2(fzgz[i + m] + dfzgz[i + m]*t[ien]);
            ee = (fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2)) - 2;
        }
        else {
            double fz2 = abs2(fzgz[i]);
            double gz2 = abs2(fzgz[i + m]);
            ee = (fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2)) - 2;
        }

        if (epow != 1)
            e += powf(fabs(2*ee), epow);          
        else
            e += fabs(2*ee);            // fabs is for avoid numerical issue when e is exactly 0
    }

    using BlockReduceT = cub::BlockReduce<double, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
    __shared__ typename BlockReduceT::TempStorage  temp_storage;

    real block_en = real( BlockReduceT(temp_storage).Sum(e) );

    if (threadIdx.x == 0) 
        atomicAdd(en+ien, block_en);
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_map_sum_dargfz(real *sum, const Complex *fz, const Complex *dfz, int m, const real *t)
{
    const int it = blockIdx.x*blockDim.x + threadIdx.x;
    const int isum = blockIdx.y;

    double s = 0;
    int s2 = 0;

#pragma unroll
    for (int j = 0; j < nItemPerReduceThread; j++) {
        int i = it*nItemPerReduceThread + j;
        if (i >= m) continue;

        const int k = (i < m - 1)?i+1:0;
        const Complex fz0 = fz[i] + dfz[i] * t[isum];
              Complex fz1 = fz[k] + dfz[k] * t[isum];

        fz1 = fz1*conj(fz0);
        double dtheta = atan2(fz1.y, fz1.x);
        s += dtheta;

        if( abs(dtheta)>2 ) ++s2;
    }

    using BlockReduceT = cub::BlockReduce<double, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
    __shared__ typename BlockReduceT::TempStorage  temp_storage;

    real block_sum = real(BlockReduceT(temp_storage).Sum(s));

    __syncthreads(); 

    real block_sum2 = real(BlockReduceT(temp_storage).Sum(s2));

    if (threadIdx.x == 0) {
        atomicAdd(sum + isum * 2,   block_sum);
        atomicAdd(sum + isum * 2+1, block_sum2);
    }
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_map_validate_vanishing_fz(int *sum, const Complex *fz, const Complex *dfz, int m,
    const Complex *fzzgzz, const Complex *dfzzgzz, const real* delta_L_fzgz, 
    const real* sampleSpacings,  const real *t, const real step0)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int istep = blockIdx.y;  // == 0

    int check1 = 0;
    if (i < m) {
        const int istep = blockIdx.y;

        const real tt = t[istep];
        const int k = (i < m - 1) ? i + 1 : 0;
        const Complex fz0 = fz[i] + dfz[i] * tt;
        Complex fz1 = fz[k] + dfz[k] * tt;

        const double delta_L_fz = delta_L_fzgz[i] + (delta_L_fzgz[i + m * 2] - delta_L_fzgz[i])*tt / step0;
        double L_fz = 0.5*(abs(fzzgzz[i] + dfzzgzz[i] * tt) + abs(fzzgzz[k] + dfzzgzz[k] * tt));
        L_fz += sampleSpacings[i] * delta_L_fz;

        const double avgAbsFz = 0.5*(abs(fz0) + abs(fz1));
        fz1 = fz1*conj(fz0);
        const double deltaTheta = abs(atan2(fz1.y, fz1.x));

        check1 = (2 + deltaTheta)*L_fz*sampleSpacings[i] > (2 - deltaTheta)*avgAbsFz;
    }

    using BlockReduceT = cub::BlockReduce<int, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
    __shared__ typename BlockReduceT::TempStorage  temp_storage;

    int block_check_sum = BlockReduceT(temp_storage).Sum(check1);

    if (threadIdx.x == 0 && block_check_sum > 0)
        atomicAdd(sum + istep, block_check_sum);
}

//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_map_distortion_bounds(real *bounds, const Complex *fzgz, const Complex *dfzgz, int m,
    const Complex *fzzgzz, const Complex *dfzzgzz, const real* delta_L_fzgz, 
    const real* sampleSpacings,  const real *t, const real step0)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    double all_bounds[6] = { 0, 0, 0, 0, 0, 0 };
    const int istep = blockIdx.y;

    if (i < m) {
        const real tt = t[istep];
        const int k = (i < m - 1) ? i + 1 : 0;
        const double absfz0 = abs(fzgz[i] + dfzgz[i]*tt);
        const double absfz1 = abs(fzgz[k] + dfzgz[k]*tt);
        const double absgz0 = abs(fzgz[i + m] + dfzgz[i + m]*tt);
        const double absgz1 = abs(fzgz[k + m] + dfzgz[k + m]*tt);
        const double avg_absfz = 0.5*(absfz0 + absfz1);
        const double avg_absgz = 0.5*(absgz0 + absgz1);

        const real* delta_L_fzgz_step0 = delta_L_fzgz + m * 2;
        double L_fz = 0.5*(abs(fzzgzz[i]     + dfzzgzz[i]     * tt) + abs(fzzgzz[k]     + dfzzgzz[k] * tt));
        double L_gz = 0.5*(abs(fzzgzz[i + m] + dfzzgzz[i + m] * tt) + abs(fzzgzz[k + m] + dfzzgzz[k + m] * tt));

        // step0 might be 0
        const real tt2 = (step0 < 1e-15) ? 0 : tt / step0;
        L_fz += sampleSpacings[i] * (delta_L_fzgz[i] +     (delta_L_fzgz_step0[i]     - delta_L_fzgz[i])    *tt2);
        L_gz += sampleSpacings[i] * (delta_L_fzgz[i + m] + (delta_L_fzgz_step0[i + m] - delta_L_fzgz[i + m])*tt2);

        const double min_absfz = avg_absfz - L_fz*sampleSpacings[i];
        const double max_absgz = avg_absgz + L_gz*sampleSpacings[i];
        const double max_absfz = avg_absfz + L_fz*sampleSpacings[i];


        //double sigma1__seg = max_absfz + max_absgz;
        //double sigma2_seg = min_absfz - max_absgz;
        //double k_seg = max_absgz / min_absfz;

        //double sigma1 = absfz0 + absgz0;
        //double sigma2 = absfz0 - absgz0;
        //double k = absgz0 / absfz0;

        //double all_bounds[] = { max_absfz + max_absgz, min_absfz - max_absgz, max_absgz / min_absfz,
        //                        absfz0 + absgz0, absfz0 - absgz0, absgz0 / absfz0 };

        all_bounds[0] = absfz0 + absgz0;
        all_bounds[1] = absfz0 - absgz0;
        all_bounds[2] = absgz0 / absfz0;
        all_bounds[3] = max_absfz + max_absgz;
        all_bounds[4] = min_absfz - max_absgz;
        all_bounds[5] = max_absgz / min_absfz;

        all_bounds[1] = exp(-all_bounds[1]);  // TODO: this is temporal hack to avoid atomicMin with negative entries which seems buggy, 
        all_bounds[4] = exp(-all_bounds[4]);
    }

    using BlockReduceT = cub::BlockReduce<double, threadsPerBlock, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    __shared__ typename BlockReduceT::TempStorage  temp_storage;

    for (int i = 0; i < 6; i++) {
        real block_bound = real( BlockReduceT(temp_storage).Reduce(all_bounds[i], cub::Max() ) );
        atomicMaxPositive(bounds + i, block_bound);

        //if (i % 3 == 1) {  // sigma2 , take min
        //    double block_bound = BlockReduceT(temp_storage).Reduce(all_bounds[i], cub::Min() );
        //    if (threadIdx.x == 0)
        //        //atomicMinPositive(bounds+i, block_bound);
        //        //atomicMin(bounds+i, block_bound);
        //}
        //else {             // sigma1 and k, take max
        //    double block_bound = BlockReduceT(temp_storage).Reduce(all_bounds[i], cub::Max());
        //    if (threadIdx.x == 0)
        //        atomicMax(bounds+i, block_bound);
        //}

        if(i<5)  __syncthreads();
    }
}







//////////////////////////////////////////////////////////////////////////
template<class Complex>
__global__ void isometry_gradient_diagscales(Complex *x, const Complex *fzgz, int m, double epow) 
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= m) return;

	const double fz2 = abs2(fzgz[i]), gz2 = abs2(fzgz[i + m]);
    using real = decltype(fzgz[i].x);

    double w = 4;
    if (epow != 1)
        w *= powf(2*fabs((fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2)) - 2), epow - 1)*epow;

	const int j = i + m;
    // fzgzb(:, 1).*(1 - ((fzgzb2*[1;-1]).^-3).*(fzgzb2*[1;3])) 
    // fzgzb(:, 2).*(1 + ((fzgzb2*[1;-1]).^-3).*(fzgzb2*[3;1]))
	x[i] = fzgz[i] * real( (1 - (fz2 + 3 * gz2) / pow3(fz2 - gz2) ) * w );
	x[j] = fzgz[j] * real( (1 + (3 * fz2 + gz2) / pow3(fz2 - gz2) ) * w );
}
 

//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void isometry_hessian_diagscales(real *const s, const Complex *fzs, const Complex *gzs, const int *samples, int m, real epow, bool spdHessian = true)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m) return;

    const int sidx = samples ? samples[i] : i;
    const Complex &fz = fzs[sidx], &gz = gzs[sidx];
    const double fz2 = abs2(fz), gz2 = abs2(gz);
    double s1 = (1 - (fz2 + 3 * gz2) / pow3(fz2 - gz2))     *4;         // *4 for hessian*4
    double s2 = (1 + (3 * fz2 + gz2) / pow3(fz2 - gz2))     *4;
    double s3 = (fz2 + 5 * gz2) / pow4(fz2 - gz2) * 4       *4;
    double s4 = (5 * fz2 + gz2) / pow4(fz2 - gz2) * 4       *4;
    double s5 = (fz2 + gz2) / pow4(fz2 - gz2) * 12          *4;

    if (epow != 1) {
        double e = 2 * fabs((fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2)) - 2);
        double w1 = powf(e, epow - 1)*epow;
        //double w2 = powf(e, epow - 2)*(epow-1)*epow;
        double w2 = min( w1*(epow-1)/e, 1e10 );  // avoid overflow, for zero energy: e = 0

        s3 = w1*s3 + w2*sqr(s1); 
        s4 = w1*s4 + w2*sqr(s2); 
        s5 = w1*s5 - w2*s1*s2; 

        s1 *= w1;
        s2 *= w1;
    }

    s5 *= -1;
    if (spdHessian && s1 < 0){
        s3 += s1 / fz2;
        s1 = 0;
    }


    real ss3     = real(fz.x * fz.y*s3);
    s[i]         = real(sqr(fz.x)*s3 + s1);
    s[i + m]     = ss3;
    s[i + m * 2] = ss3;
    s[i + m * 3] = real(sqr(fz.y)*s3 + s1);


    i += m * 4;
    ss3          = real(gz.x * gz.y * s4);
    s[i]         = real(sqr(gz.x)*s4 + s2);
    s[i + m]     = ss3;
    s[i + m * 2] = ss3;
    s[i + m * 3] = real(sqr(gz.y)*s4 + s2);

    i += m * 4;
    s[i]         = real(fz.x * gz.x * s5);
    s[i + m]     = real(fz.x * gz.y * s5);
    s[i + m * 2] = real(fz.y * gz.x * s5);
    s[i + m * 3] = real(fz.y * gz.y * s5);
}

//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void maxtForBoundk(real *t, const Complex * fz, const Complex *gz, const Complex *dfz, const Complex *dgz, int m, double k2) 
{
	const int it = blockIdx.x*blockDim.x + threadIdx.x;

    double tt = CUDART_INF;			 //CUDART_INF_F;  // for single precision

#pragma unroll
    for (int j = 0; j < nItemPerReduceThread; j++) {
        int i = it*nItemPerReduceThread + j;
        if (i >= m) continue;

        double a = k2*abs2(dfz[i]) - abs2(dgz[i]);
        double b = 2 * (k2* dot(fz[i], dfz[i]) - dot(gz[i], dgz[i]));
        double c = k2*abs2(fz[i]) - abs2(gz[i]);
        double delta = b*b - 4 * a*c;

        const double scale_factor = 0.8;
        if (!(a > 0 && (delta < 0 || b>0)))
            tt = min(tt,  max(0., scale_factor*(-b - sqrt(delta)) / a / 2) );    // bug fixed, may produced -0.0, which is problematic for the atomicMin hack later
            //tt = min(tt,  (-b - sqrt(delta)) / a / 2 );   
    }


    using BlockReduceT = cub::BlockReduce<double, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
    __shared__ typename BlockReduceT::TempStorage  temp_storage;

    real block_min = real( BlockReduceT(temp_storage).Reduce(tt, cub::Min()) );

    // https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html
    // A subsequent __syncthreads() threadblock barrier should be invoked after calling this method if the 
    // collective's temporary storage (e.g., temp_storage) is to be reused or repurposed.

    if (threadIdx.x == 0) 
        atomicMinPositive(t, block_min);            // todo: atomicMin for float?  the current code works only if all numbers are non-negative
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__forceinline__ __host__ __device__ 
void p2p_harmonic_full_energy(const Complex *fzgz, const Complex *dfzgz, const Complex *fgP2P, const Complex *dfgP2P, const Complex* bP2P, int m, int nP2P, 
    const real *steps, real *en, real isoEPow, real lambda, bool softP2P, int nsteps = 1, real *en_isometry = nullptr)
{
    isometry_energy <<<dim3(reduceBlockNum(m), nsteps), threadsPerBlock >>> (en, fzgz, dfzgz, steps, m, isoEPow);
    CUDA_CHECK_ERROR;

    print_gpu_value(en, "isometric energy", false);
    
    // save isometric energies separately, for potential display/debug later 
    if(en_isometry) cudaMemcpyAsync(en_isometry, en, sizeof(real)*nsteps, cudaMemcpyDeviceToDevice);

    // add P2P energy, if apply
    if (softP2P) {
        p2p_energy <<<dim3(blockNum(nP2P), nsteps), threadsPerBlock >>> (en, fgP2P, dfgP2P, steps, bP2P, nP2P, lambda);
        CUDA_CHECK_ERROR;

        print_gpu_value(en, "total energy");
    }
};

//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_line_search(const Complex *fzgz, const Complex *dfzgz, const Complex *fgP2P, const Complex *dfgP2P, const Complex* bP2P, int m, int nP2P,
    real *steps, real *e, real isoepow, const real *dgdotdpp_and_norm2dpp, real lambda, bool softP2P, int enEvalsPerKernel)
{
    const double ls_alpha = 0.2;
    const double ls_beta = 0.5;

    const real dgdotfz = *dgdotdpp_and_norm2dpp;
    const real normdpp = sqrt(dgdotdpp_and_norm2dpp[1]);

    //real en[1];
    real *const en = e + 1;
    real *const sum_dargfz = en + enEvalsPerKernel*2;
    auto fQPEstim = [&](real t) { return *e + ls_alpha*t*dgdotfz; };

#if 1 // one thread for linesearch
    if (threadIdx.x > 0) return;

    bool done = false;
    while (!done){
        for (int i = 1; i < enEvalsPerKernel; i++) steps[i] = *steps / (1 << i); // powf(ls_beta, i);

        //////////////////////////////////////////////////////////////////////////
        //for (int i = 0; i < enEvalsPerKernel; i++) en[i] = 0;
        cudaMemsetAsync(en, 0, sizeof(real)*enEvalsPerKernel);
#ifdef _DEBUG
        p2p_harmonic_full_energy(fzgz, dfzgz, fgP2P, dfgP2P, bP2P, m, nP2P, steps, en, isoepow, lambda, softP2P, enEvalsPerKernel, en + enEvalsPerKernel);
        cudaDeviceSynchronize();
        for (int i = 0; i < enEvalsPerKernel; i++) {
            gpu_print(en[i + enEvalsPerKernel], "isometric energy", false);
            gpu_print(en[i], "total energy");
        }
#else
        p2p_harmonic_full_energy(fzgz, dfzgz, fgP2P, dfgP2P, bP2P, m, nP2P, steps, en, isoepow, lambda, softP2P, enEvalsPerKernel);
#endif

        //////////////////////////////////////////////////////////////////////////
        // approximate argument principal to make sure fz does not vanish inside
        cudaMemsetAsync(sum_dargfz, 0, sizeof(real)*enEvalsPerKernel*2);
        harmonic_map_sum_dargfz <<<dim3(reduceBlockNum(m), enEvalsPerKernel), threadsPerBlock >>> (sum_dargfz, fzgz, dfzgz, m, steps);
        CUDA_CHECK_ERROR;
        cudaDeviceSynchronize();

        for (int i = 0; i < enEvalsPerKernel; i++) {
            real ls_t = steps[i];
            const real argment_principal_eps = 1;
            if (ls_t*normdpp < minDeformStepNorm || (en[i] < fQPEstim(ls_t) && sum_dargfz[i * 2] < argment_principal_eps && sum_dargfz[i * 2 + 1] < 0.2)) {
                steps[0] = ls_t;
                *e = en[i]; // output energy
                done = true;
                break;
            }
        }
        if (!done) steps[0] /= (1<<enEvalsPerKernel);
        gpu_print(*steps, "step size");
    }

    //if (*steps*normdpp < minDeformStepNorm) *steps = 0; // avoid the updating later in optimization
    gpu_print(*steps, "step size");

#else // use multiple threads for line search
    const int i = threadIdx.x;
    if (i > 0) steps[i] = steps[0] * pow(ls_beta, i);

    __shared__ int imaxStepSize;

    while (true){
        if (i == 0) {
            cudaMemsetAsync(en, 0, sizeof(real)*enEvalsPerKernel);
#ifdef _DEBUG
            p2p_harmonic_full_energy(fzgz, dfzgz, fgP2P, dfgP2P, bP2P, m, nP2P, steps, en, isoepow, lambda, softP2P, enEvalsPerKernel, en + enEvalsPerKernel);
            cudaDeviceSynchronize();
            for (int i = 0; i < enEvalsPerKernel; i++) {
                gpu_print(en[i + enEvalsPerKernel], "isometric energy", false);
                gpu_print(en[i], "total energy");
            }
#else
            p2p_harmonic_full_energy(fzgz, dfzgz, fgP2P, dfgP2P, bP2P, m, nP2P, steps, en, isoepow, lambda, softP2P, enEvalsPerKernel);
#endif

            //////////////////////////////////////////////////////////////////////////
            // approximate argument principal to make sure fz does not vanish inside
            cudaMemsetAsync(sum_dargfz, 0, sizeof(real)*enEvalsPerKernel * 2);
            harmonic_map_sum_dargfz <<<dim3(reduceBlockNum(m), enEvalsPerKernel), threadsPerBlock >>> (sum_dargfz, fzgz, dfzgz, m, steps);
            CUDA_CHECK_ERROR;
            cudaDeviceSynchronize();

            imaxStepSize = 100;
        }

        __syncthreads();

        real ls_t = steps[i];
        const real argment_principal_eps = 1;
        if (ls_t*normdpp < minDeformStepNorm || (en[i] < fQPEstim(ls_t) && sum_dargfz[i * 2] < argment_principal_eps && sum_dargfz[i * 2 + 1] < 0.2)) {
            atomicMin(&imaxStepSize, i);
        }

        if (imaxStepSize < enEvalsPerKernel) break;

        steps[i] *= pow(ls_beta, enEvalsPerKernel);
        if(i==0) gpu_print(*steps, "step size");
    }

    if (i == 0 && imaxStepSize < enEvalsPerKernel) {
        steps[0] = steps[imaxStepSize];
    }

    if (i == 0 && *steps*normdpp < minDeformStepNorm) *steps = 0; // avoid the updating later in optimization
    if (i == 0) gpu_print(*steps, "step size");
#endif
}



//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_map_validate_bounds(const Complex *fzgz, const Complex *dfzgz, int m, const Complex *fzzgzz, const Complex *dfzzgzz, 
    const real *delta_L_fzgz, const real* sampleSpacings, real *ls_t, real *bounds, const real *norm2dpp, int enEvalsPerKernel)
{
    const double ls_beta = 0.5;
    const real normdpp = sqrt(*norm2dpp);

    int *const vanishing_fz_check_sum = (int*)bounds;
    const real step0 = *ls_t;
    while (*ls_t*normdpp > minDeformStepNorm) {
        *vanishing_fz_check_sum = 0;
        harmonic_map_validate_vanishing_fz <<<blockNum(m), threadsPerBlock >>> (vanishing_fz_check_sum, fzgz, dfzgz, m,
            fzzgzz, dfzzgzz, delta_L_fzgz, sampleSpacings, ls_t, step0);

        cudaDeviceSynchronize();

#ifdef _DEBUG
        printf("%20s = %d\n", "vanishing_fz_check_sum:", *vanishing_fz_check_sum);
#endif

        if (*vanishing_fz_check_sum == 0)  // fz non vanishing inside
            break;

        *ls_t *= ls_beta;
    }

    gpu_print(*ls_t, "step size (non-vanishing fz)");


    // need at least one run for the distortion bounds, TODO: or just report that it has converged
    do{        
        cudaMemsetAsync(bounds, 0, sizeof(real)*6);
        harmonic_map_distortion_bounds <<<blockNum(m), threadsPerBlock >>> (bounds, fzgz, dfzgz, m, fzzgzz, dfzzgzz, delta_L_fzgz, sampleSpacings, ls_t, step0);

        cudaDeviceSynchronize();

        bounds[1] = -log(bounds[1]);
        bounds[4] = -log(bounds[4]);
#ifdef _DEBUG
        printf("distortions = sigma1: (%.3f, %.3f) sigma2: (%.3f, %.3f) k: (%.3f, %.3f)\n", bounds[0], bounds[3], bounds[1], bounds[4], bounds[2], bounds[5]);
#endif
        if (bounds[4] >= 0) // sigma2_lower_bound
            break;

        *ls_t *= ls_beta;
    } while (*ls_t*normdpp > minDeformStepNorm);

    gpu_print(*ls_t, "step size (sigma2>0)");

    if (*ls_t*normdpp < minDeformStepNorm) *ls_t = 0; // avoid the updating later in optimization
}


#define ChooseCublasFunc(fun, dfun, sfun) \
    using fun##type = std::conditional_t<runDoublePrecision, decltype(&cublas##dfun), decltype(&cublas##sfun)>; \
    const auto fun = runDoublePrecision ? (fun##type)&cublas##dfun : (fun##type)&cublas##sfun;

#define ChooseCusolverDnFunc(fun, dfun, sfun) \
    using fun##type = std::conditional_t<runDoublePrecision, decltype(&cusolverDn##dfun), decltype(&cusolverDn##sfun)>; \
    const auto fun = runDoublePrecision ? (fun##type)&cusolverDn##dfun : (fun##type)&cusolverDn##sfun;

template<class Complex, class real>
struct cuHarmonicMap
{
    using vecC = cuVector<Complex>;
    using vecR = cuVector<real>;


	vecC phipsy;    // phi and psi

	const Complex *D2;
	const int *hessian_samples = nullptr;
	vecR h;	   // Hessian
	vecC g;    // gradient  // isometry or p2p or combined, size 2n+1, last entry always set to 0, for AQP optimization
	vecR gR;   // gradient in real
	int m;	   // number of sample, #row in D2
	int n;	   // number of cage vertex, #column in D2
    int mh;    // number of sample for hessian, can be <= m, the first mh rows of D2 is used for hessian computation

    real isoepow = 1.; // raise isometry energy to power

	vecC fzgz;
	vecR DRDI;

	cublasHandle_t cub_hdl;
	vecC constants;
    Complex *pOne = nullptr, *pZero = nullptr, *pMinusOne = nullptr;
    Complex *p2Lambda = nullptr, *pHessianSampleRate = nullptr;        // constants 


    // whether/how to modify the hessian of the isometry energy to be SPD
    // 0: no modification vanilla hessian, 1: per-sample SPD hessian, 2: SPD hessian using eigenfactorization on hessian of the full mapping
    enum { ISO_HESSIAN_VANILLA = 0, ISO_HESSIAN_SPD_PERSAMPLE = 1, ISO_HESSIAN_SPD_FULLEIG=2 };
    int modifyHessianSPD = ISO_HESSIAN_SPD_PERSAMPLE; 


    vecR h_tmpstorage; // for computing hessian
    vecR h_diagscales; // for computing hessian / gradient



	const Complex *C2;      // Cauchy coord at p2p 
    int nP2P;
    vecC  fgP2P;
    vecC phipsy_tmpstorage, bP2P_tmpstorage, matGradP2P;
    real P2P_weight;

    vecR C2C2BarReal;

    cuCholSolverDn<real> solverCHOL;
    cuLUSolverDn<real> solverLU;

    int linearSolvePref = LINEAR_SOLVE_PREFER_CHOLESKY;
    real deltaFixSPDH = real(1e-15);

    static const bool runDoublePrecision = std::is_same<real, double>::value;


	cuHarmonicMap(const Complex *d_D2, int nSample, int dim, const int *hessian_sample_indices, int nHessSample) 
        :D2(d_D2), m(nSample), n(dim), hessian_samples(hessian_sample_indices), mh(nHessSample), phipsy(dim*2), 
        g(dim*2+1), gR(dim*4), h(dim*dim*16), cub_hdl(0), C2(nullptr), nP2P(0), P2P_weight(0),
        solverCHOL(dim*4-2), solverLU(dim*4) 
    {myZeroFill(g.data() + n * 2); }
    ~cuHarmonicMap() { cublasDestroy(cub_hdl); }

    void init();

    void setupP2P(const Complex *d_C2, int numberP2P, real lambda);

    void update_bP2P(const Complex *bP2P) {   // bP2P: target positions of p2p constraints, update matGradP2P (last column) for p2p gradient evaluation
        ChooseCublasFunc(axpy, Daxpy, Saxpy);
        auto consts = std::vector<Complex>( constants );

        myZeroFill(matGradP2P.data() + nP2P*n * 2, nP2P);
        cublasStatus_t sta = axpy(cub_hdl, nP2P * 2, (const real*)pMinusOne, (const real*)bP2P, 1, (real*)( matGradP2P.data() + nP2P*n*2 ), 1);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
    }
    

    void update_phipsy(const Complex *d_phipsy) {
        myCopy_n(d_phipsy, n * 2, phipsy.data());
    }

	void increment_phipsy(const Complex *dphipsy, const real* step) {
		// phipsy += dphipsy*step
        ChooseCublasFunc(axpy, Daxpy, Saxpy);
        cublasStatus_t sta = axpy(cub_hdl, n * 4, step, (const real*)(dphipsy), 1, (real*)phipsy.data(), 1);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
	}

    void update_fzgz(Complex *const pfzgz = nullptr, const Complex *pphipsy = nullptr) {
		// fzgz = D2*phipsy
        ChooseCublasFunc(cgemm, Zgemm, Cgemm);
		cublasStatus_t sta = cgemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, m, 2, n, pOne, D2, m, 
                                   pphipsy?pphipsy:phipsy.data(), n, pZero, pfzgz?pfzgz:fzgz.data(), m);
		ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
	}

	void increment_fzgz(const Complex *dfzgz, const real* step) {
		// fzgz += dfzgz*step
        ChooseCublasFunc(axpy, Daxpy, Saxpy);
        cublasStatus_t sta = axpy(cub_hdl, m * 4, step, (const real*)(dfzgz), 1, (real*)fzgz.data(), 1);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
	}

    void update_fgP2P(Complex *const pfgP2P = nullptr, const Complex *pphipsy = nullptr) {
        // fgP2P = C2*phipsy  nP2P*n * n*2
        ChooseCublasFunc(cgemm, Zgemm, Cgemm);
        cublasStatus_t sta = cgemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, nP2P, 2, n, pOne, C2, nP2P, 
                                   pphipsy?pphipsy:phipsy.data(), n, pZero, pfgP2P?pfgP2P:fgP2P.data(), nP2P);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
    }

	void increment_fgP2P(const Complex *dfgP2P, const real* step) {
		// fgP2P += dfgP2P*step
        ChooseCublasFunc(axpy, Daxpy, Saxpy);
        cublasStatus_t sta = axpy(cub_hdl, nP2P * 4, step, (const real*)(dfgP2P), 1, (real*)fgP2P.data(), 1);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
	}

    void compute_max_step_to_preserve_orientation(real* step, const Complex* dfzgz, real kmax2 = 1.) {
        // by default set step=1, used as input for atomicMin, 
        maxtForBoundk<<< reduceBlockNum(m), threadsPerBlock >>> (step, fzgz.data(), fzgz.data() + m, dfzgz, dfzgz + m, m, kmax2);
        CUDA_CHECK_ERROR;
    }
 
    const Complex* isometry_gradient(Complex* pG = nullptr) {
        assert(h_diagscales.size() >= m * 4);
        Complex *gradscales = (Complex*)(h_diagscales.data());      // diagscales has memory >= 2m complex number or 4m real numbers
		isometry_gradient_diagscales<<<blockNum(m), threadsPerBlock>>>(gradscales, fzgz.data(), m, isoepow);
		CUDA_CHECK_ERROR;

        pG = pG?pG:g.data();
        ChooseCublasFunc(cgemm, Zgemm, Cgemm);

		// D2'*() = grad
        cublasStatus_t sta = cgemm(cub_hdl, CUBLAS_OP_C, CUBLAS_OP_N, n, 2, m, pOne, D2, m, gradscales, m, pZero, pG, n);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        return pG;
    }

    const real* isometry_gradientReal() {
        isometry_gradient();
        real *const pGR = gR.data();
        vectorComplex2Real<<<blockNum(n*2), threadsPerBlock>>>(g.data(), n*2, pGR);
        return pGR;
    }

    const real* isometry_hessian(real* hess = nullptr);

    const real* hessian_full(real* hess = nullptr) {
        isometry_hessian(hess);

        if (modifyHessianSPD != ISO_HESSIAN_VANILLA &&  deltaFixSPDH > 0)
            addToMatrixDiagonal<<<blockNum(n*4), threadsPerBlock>>>(hess, deltaFixSPDH*P2P_weight, n*4);

        ChooseCublasFunc(gemm, Dgemm, Sgemm);
        ChooseCublasFunc(geam, Dgeam, Sgeam);

        hess = hess ? hess : h.data();
        const int n2 = n * 2;
        //////////////////////////////////////////////////////////////////////////
        // flip sign for dpsibar
        cublasStatus_t sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, n * 3, n, (real*)pMinusOne, hess + n*n * 12, n2 * 2, (real*)pZero, hess, n2 * 2, hess + n*n * 12, n2 * 2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, n, n * 3, (real*)pMinusOne, hess + n * 3, n2 * 2, (real*)pZero, hess, n2 * 2, hess + n * 3, n2 * 2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");


        //////////////////////////////////////////////////////////////////////////
        // add p2p grad
        const real *d_CCBar = C2C2BarReal.data();        // [C2 conj(C2) -bP2P]
        // compensate for sparse sampling for the hessian
        sta = gemm(cub_hdl, CUBLAS_OP_T, CUBLAS_OP_N, n2*2, n2*2, nP2P*2, (real*)p2Lambda, d_CCBar, nP2P*2, d_CCBar, nP2P*2, (real*)pHessianSampleRate, hess, n2*2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        return hess;
    }

    void p2p_gradient(Complex *const grad, bool computeP2PdiffOnly = false) {
        // add gradient of p2p energy to grad or compute P2Pdiff for hard constraints method
        const Complex *d_matGradP2P = matGradP2P.data();        // [C2 conj(C2) -bP2P]
        Complex *const d_phipsy_tmp = phipsy_tmpstorage.data();
        Complex *const p2pDiff = computeP2PdiffOnly?grad:bP2P_tmpstorage.data();

        const int n2 = n * 2;
        myCopy_n(phipsy.data(), n2, d_phipsy_tmp);

        conjugate_inplace <<<blockNum(n), threadsPerBlock>>> (d_phipsy_tmp + n, n);

        ChooseCublasFunc(cgemm, Zgemm, Cgemm);

        // p2pError= C2*phi+conj(C2*psy) - bP2P
        cublasStatus_t sta = cgemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, nP2P, 1, n2 + 1, pOne, d_matGradP2P, nP2P, d_phipsy_tmp, n2 + 1, pZero, p2pDiff, nP2P);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        if (!computeP2PdiffOnly) {
            sta = cgemm(cub_hdl, CUBLAS_OP_C, CUBLAS_OP_N, n2, 1, nP2P, p2Lambda, d_matGradP2P, nP2P, p2pDiff, nP2P, pOne, grad, n2);
            ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        }
    }


    void computeNewtonStep(Complex *const dpp, Complex *const grad, int SPDHessian = ISO_HESSIAN_SPD_PERSAMPLE, cudaEvent_t *times=nullptr); 
};

template<class Complex, class real>
void cuHarmonicMap<Complex, real>::setupP2P(const Complex *d_C2, int numberP2P, real lambda) 
{
    C2 = d_C2;
    nP2P = numberP2P;
    P2P_weight = lambda;

    fgP2P.resize(nP2P * 2);

    lambda *= 2;            // for p2p gradient evaluation only, therefore multiply by 2
    myCopy_n(&lambda, 1, (real*)p2Lambda, cudaMemcpyHostToDevice);


    bP2P_tmpstorage.resize(nP2P);

    const int n2 = n * 2;
    phipsy_tmpstorage.resize(n2 + 1);  // [phi; conj(psy); 1]
    myCopy_n(pOne, 1, phipsy_tmpstorage.data() + n2);

    matGradP2P.resize(nP2P*(n2 + 1));   // grad = 2*[C2 conj(C2)]' * [C2 conj(C2) -bP2P] * [phi; conj(psy); 1]   size: nP2P*(2n+1)   // [C2 conj(C2)]' size: (2n*nP2P)
    myCopy_n(d_C2, nP2P*n, matGradP2P.data());
    myCopy_n(d_C2, nP2P*n, matGradP2P.data() + nP2P*n);
    conjugate_inplace <<<blockNum(nP2P*n), threadsPerBlock >>> (matGradP2P.data() + nP2P*n, nP2P*n);


    C2C2BarReal.resize(nP2P*n2 * 4);        // for computing full hessian
    matrixRowsComplex2Real <<<dim3(ceildiv(nP2P, 4), ceildiv(n2, 32)), dim3(4, 32) >>> (matGradP2P.data(), nP2P, n2, C2C2BarReal.data());
}

template<class Complex, class real>
void cuHarmonicMap<Complex, real>::init() 
{
    ensure(CUBLAS_STATUS_SUCCESS == cublasCreate(&cub_hdl), "CUBLAS initialization failed");
    cublasSetPointerMode(cub_hdl, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetAtomicsMode(cub_hdl, CUBLAS_ATOMICS_ALLOWED);

    //////////////////////////////////////////////////////////////////////////
    const std::vector<Complex> consts_cpu = { { 1, 0 }, { 0, 0 }, { -1, 0 }, {1, 0}, {m / real(mh), 0} };
    constants = consts_cpu;

    pOne = constants.data();
	pZero = pOne + 1;
    pMinusOne = pOne + 2;
    p2Lambda = pOne + 3;
    pHessianSampleRate = pOne + 4;

    //////////////////////////////////////////////////////////////////////////
    fzgz.resize(m * 2);

    //////////////////////////////////////////////////////////////////////////
    DRDI.resize(mh*n * 4);
    matrixRowsComplex2Real <<<dim3(ceildiv(mh, 32), ceildiv(n, 32)), dim3(32, 32) >>> (D2, m, n, DRDI.data(), hessian_samples, mh);
    CUDA_CHECK_ERROR;

    h_tmpstorage.resize(mh * n * 8);
    h_diagscales.resize( std::max(mh * 4 * 3, m*4) );   // make sure enough memory, which is shared by grad (complex) and hessian (real) evaluation!
}

template<class Complex, class real>
const real* cuHarmonicMap<Complex, real>::isometry_hessian(real* hess) 
{
    const real *d_DRDI = DRDI.data();

    const real *pOneR = (real*)pOne;
    const real *pZeroR = (real*)pZero;
    const real *pMinusOneR = (real*)pMinusOne;

    ChooseCublasFunc(rgemm, Dgemm, Sgemm);
    ChooseCublasFunc(dgmm, Ddgmm, Sdgmm);
    ChooseCublasFunc(geam, Dgeam, Sgeam);
    //ChooseCublasFunc(syrk, Dsyrk, Ssyrk)

    const int mh2 = mh * 2;
    const int n2 = n * 2;
    const int n4 = n * 4;

    if (n2 > mh) h_tmpstorage.resize(mh2*n2 + n2*n2 * 2);  // for the optimized matrix loading in gemm below. original size 8mn: 4mn+8n*n=4n(m+2n): 2n<m
    real *const d_Htmp1 = h_tmpstorage.data();

    //////////////////////////////////////////////////////////////////////////
    auto ComputeSubHessian = [=](const real* ss, real* subH) {
        real *const d_Htmp2 = d_Htmp1 + mh2*n2;
        cublasStatus_t sta = dgmm(cub_hdl, CUBLAS_SIDE_LEFT, mh2, n2, d_DRDI, mh2, ss, 1, d_Htmp1, mh2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        sta = dgmm(cub_hdl, CUBLAS_SIDE_LEFT, mh2, n2, d_DRDI, mh2, ss + mh2, 1, d_Htmp2, mh2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        // output to d_Htmp1
        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, mh, n2, pOneR, d_Htmp1, mh2, pOneR, d_Htmp1 + mh, mh2, d_Htmp1, mh2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, mh, n2, pOneR, d_Htmp2, mh2, pOneR, d_Htmp2 + mh, mh2, d_Htmp1 + mh, mh2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        // test: syrk is much less optimized!
        //sta = syrk(cub_hdl, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n2, mh2, pOneR, d_DRDI, mh2, pZeroR, subH, n2 * 2);

#if 1
        // reduce matrix A loading time TODO: make sure d_Htmp2 has enough memory >2n*4n!
        sta = rgemm(cub_hdl, CUBLAS_OP_T, CUBLAS_OP_N, n2, n4, mh, pOneR, d_DRDI, mh2, d_Htmp1, mh, pZeroR, d_Htmp2, n2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        //sta = rgemm(cub_hdl, CUBLAS_OP_T, CUBLAS_OP_N, n2, n2, mh, pOneR, d_DRDI, mh2, d_Htmp1 + mh, mh2, pZeroR, d_Htmp2 + n2, n2 * 2);
        //ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, n, n2, pOneR, d_Htmp2, n4, pMinusOneR, d_Htmp2 + n*3, n4, subH, n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, n, n2, pOneR, d_Htmp2 + n, n4, pOneR, d_Htmp2 + n*2, n4, subH + n, n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
#else
        sta = rgemm(cub_hdl, CUBLAS_OP_T, CUBLAS_OP_N, n2, n2, mh2, pOneR, d_DRDI, mh2, d_Htmp1, mh2, pZeroR, subH, n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
#endif
    };


    real *const diagscales = h_diagscales.data();
    hess = hess ? hess : h.data();

    isometry_hessian_diagscales <<<blockNum(mh), threadsPerBlock >>> (diagscales, fzgz.data(), fzgz.data()+m, hessian_samples, mh, isoepow, modifyHessianSPD==ISO_HESSIAN_SPD_PERSAMPLE);
    CUDA_CHECK_ERROR;

    ComputeSubHessian(diagscales, hess);
    ComputeSubHessian(diagscales + mh * 4, hess + n2*n2 * 2 + n2);
    ComputeSubHessian(diagscales + mh * 8, hess + n2*n2 * 2);

    cublasStatus_t sta = geam(cub_hdl, CUBLAS_OP_T, CUBLAS_OP_N, n2, n2, pOneR, hess + n2*n2 * 2, n4, pZeroR, hess, n4, hess + n2, n4);
    ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

    dim3 blkdim(32, 32);   // DO NOT use big block
    swapMatrixInMemory<<<dim3(ceildiv(n4, 32), ceildiv(n, 32)), blkdim>>>(hess + n*n * 4, hess + n*n * 8, n4, n, n4);
    CUDA_CHECK_ERROR;
    swapMatrixInMemory<<<dim3(ceildiv(n, 32), ceildiv(n4, 32)), blkdim>>>(hess + n, hess + n2, n, n4, n4);
    CUDA_CHECK_ERROR;




    if (modifyHessianSPD == ISO_HESSIAN_SPD_FULLEIG) {
        const bool runDoublePrecision = std::is_same<real, double>::value;

        ChooseCusolverDnFunc(syevd_bufferSize, Dsyevd_bufferSize, Ssyevd_bufferSize);
        ChooseCusolverDnFunc(syevd, Dsyevd, Ssyevd);

        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.

        cuVector<real> D(n4);

        cusolverDnHandle_t &hdl = solverCHOL.handle;
        if (!hdl) cusolverDnCreate(&hdl);

        int bufferSize = 0;
        // step 1: query working space of syevd
        syevd_bufferSize(hdl, jobz, uplo, n4, hess, n4, D.data(), &bufferSize);
        cuVector<real> buffer(bufferSize);
        cuVector<int> info(1);

        //auto H0_h = cuMem2Vec(hess, n4*n4);
        // step 2: eig
        syevd(hdl, jobz, uplo, n4, hess, n4, D.data(), buffer.data(), bufferSize, info.data());

        //auto D0_h = cuMem2Vec(D.data(), n4);
        // step 3: take sqrt of D: eigen values
        sqrt_non_negative_clamp<real> <<<blockNum(n4), threadsPerBlock>> > (D.data(), D.data(), n4);

        //auto V_h = cuMem2Vec(hess, n4*n4);
        //auto D_h = cuMem2Vec(D.data(), n4);

        cuVector<real> sqrtHess(n4*n4);
        sta = dgmm(cub_hdl, CUBLAS_SIDE_RIGHT, n4, n4, hess, n4, D.data(), 1, sqrtHess.data(), n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        sta = rgemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_T, n4, n4, n4, pOneR, sqrtHess.data(), n4, sqrtHess.data(), n4, pZeroR, hess, n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        //auto sqrtHess_h = cuMem2Vec(sqrtHess.data(), n4*n4);
        //auto H_h = cuMem2Vec(hess, n4*n4);
    }


    return hess;
}


template<class Complex, class real>
void cuHarmonicMap<Complex, real>::computeNewtonStep(Complex *const dpp, Complex *const grad_out, int SPDHessian, cudaEvent_t *times) 
{
    modifyHessianSPD = SPDHessian;

    const int n2 = n * 2;

    auto recordEvent = [times](cudaEvent_t& t) { if (times) cudaEventRecord(t); };

    update_fzgz();

    Complex *const grad = g.data();
    isometry_gradient(grad);

    conjugate_inplace <<<blockNum(n), threadsPerBlock >>> (grad + n, n);
    CUDA_CHECK_ERROR;

    if (P2P_weight > 0)  p2p_gradient(grad);
    
    myCopy_n(grad, n2, grad_out);

    recordEvent(times[0]); // record gradient time
    
    real* const hess = h.data();
    hessian_full(hess);

    recordEvent(times[1]); // record hessian time

    //////////////////////////////////////////////////////////////////////////
    real *const gradR = gR.data();

    if ( linearSolvePref==LINEAR_SOLVE_FORCE_CHOLESKY || (SPDHessian!=ISO_HESSIAN_VANILLA&&linearSolvePref==LINEAR_SOLVE_PREFER_CHOLESKY) ) {
        if (SPDHessian == ISO_HESSIAN_VANILLA) 
            fprintf(stderr, "The vanilla Hessian for isometric energy is not necessary SPD, Cholesky solver in Newton may fail!, Consider switch to LU solver!");

        solverCHOL.init(); // allocate memory first;
        squareMatrixCopyWithSkip << <dim3(ceildiv(n2 * 2 - 2, 32), ceildiv(n2 * 2 - 2, 32)), dim3(32, 32) >> > (hess, n2 * 2, n2 * 2 - 2, n2 - 1, solverCHOL.A.data());
        vectorComplex2Real << <blockNum(n2), threadsPerBlock >> > (grad, n2 - 1, gradR);


        int solver_stat = solverCHOL.factor();

        assert(solver_stat == 0);
        solverCHOL.solve(gradR);

        vectorReal2Complex << <blockNum(n2 - 1), threadsPerBlock >> > (gradR, n2 - 1, dpp, real(-1));
        myCopy_n(pZero, 1, dpp + n2 - 1);
    }
    else {
        ChooseCublasFunc(geam, Dgeam, Sgeam);
        // M([1 2n+1], :) = fC2Rm(sparse(1, 1 + n, 1, 1, 2 * n));
        cublasStatus_t sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, 1, n2 * 2, (real*)pZero, hess, n2 * 2, (real*)pZero, hess, n2 * 2, hess, n2 * 2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, 1, n2 * 2, (real*)pZero, hess, n2 * 2, (real*)pZero, hess, n2 * 2, hess + n2, n2 * 2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        //myCopy_n((const real*)pOne, 1, hess + n * 4 * n);
        //myCopy_n((const real*)pOne, 1, hess + n * 4 * n * 3 + n2);
        myCopy_n((const real*)pOne, 1, hess + n * 4 * (n2 - 1));
        myCopy_n((const real*)pOne, 1, hess + n * 4 * (n2 * 2 - 1) + n2);



        // following does not work, because matrix M and rhs vector grad should be in correspondence
        // M([2n 4n], :) = fC2Rm(sparse(1, 1 + n, 1, 1, 2 * n));
        //cublasStatus_t sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, 1, n2 * 2, (real*)pZero, hess, n2 * 2, (real*)pZero, hess, n2 * 2, hess+n2-1, n2 * 2);
        //ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        //sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, 1, n2 * 2, (real*)pZero, hess, n2 * 2, (real*)pZero, hess, n2 * 2, hess+n2*2-1, n2 * 2);
        //ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        //myCopy_n((const real*)pOne, 1, hess + n*n*4+n2-1);
        //myCopy_n((const real*)pOne, 1, hess + n*n*12+n*4-1);


        // grad[0] -> 0;
        myZeroFill(grad);
        vectorComplex2Real <<<blockNum(n2), threadsPerBlock >>> (grad, n2, gradR);
        int solver_stat = solverLU.factor(hess);
        assert(solver_stat == 0);
        solverLU.solve(gradR);

        vectorReal2Complex <<<blockNum(n2), threadsPerBlock >>> (gradR, n2, dpp, real(-1));
    }

    recordEvent(times[2]); // record solve time
}


#undef ChooseCusolverDnFunc
#undef ChooseCublasFunc
