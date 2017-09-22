#include "utils.cuh"
#include <math_constants.h>

#include <cub/block/block_reduce.cuh>

enum LINEAR_SOLVE_PREFERENCE { LINEAR_SOLVE_PREFER_CHOLESKY = 0, LINEAR_SOLVE_PREFER_LU = 1, LINEAR_SOLVE_FORCE_CHOLESKY = 2 };
enum ISOMETRIC_ENERGY_TYPE { ISO_ENERGY_SYMMETRIC_DIRICHLET = 0, ISO_ENERGY_EXP_SD = 1, ISO_ENERGY_AMIPS = 2 };

const char* IsometricEnergyNames[] = { "SymmDirichlet", "Exp Iso", "AMIPS" };
__device__ const bool SPD_hessian_modification_is_simple[] = { true, true, true, false };

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
struct CageVertexIndexOffsets {
    enum { maxHarmonicMapCageNumber = 100 };
    int n;  // number of cages
    int offsets[maxHarmonicMapCageNumber+1];
};
__constant__ CageVertexIndexOffsets cageOffsets_g;


template<class Complex, class real>
__global__ void abs_diff_similarity_polygon(real *dSimlarity, const Complex *phi, const Complex *dphi, const Complex *v, int n, const real *t)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    // process phi/psy in one thread to avoid slight complex calculation of next/prev vertex indices

#if 1  // for multiply connected domains
    int nCage = cageOffsets_g.n;
    const int *cageOffsets = cageOffsets_g.offsets;

    int npp = cageOffsets[nCage] + nCage - 1;

    if (i >= cageOffsets[nCage] + (nCage - 1) * 2) {   // log |z-rho| term
        const int j = i-(nCage - 1) * 2;
        dSimlarity[i] = abs(phi[j]);
        dSimlarity[i+n] = abs(phi[j+npp]);
        dSimlarity[i+n*2] = abs(dphi[j]**t + phi[j]);
        dSimlarity[i+n*3] = abs(dphi[j+npp]**t + phi[j+npp]);
        return;
    }

    // find which cage is the current virtual vertex on
    int icage = 0;
    while ( icage<nCage && i >= cageOffsets[icage+1] + icage * 2 ) ++icage;
    assert(icage < nCage);

    int fullOffset = cageOffsets[icage] + max(icage-1, 0) * 2;
    int fullOffsetNext = cageOffsets[icage+1] + icage * 2;
    const int next = (i+1<fullOffsetNext)?i+1:fullOffset;
    const int prev = (i>fullOffset)?i-1:fullOffsetNext-1;

    auto PPat = [icage, fullOffset](const Complex *pp, int idx) { 
        if (icage == 0) return pp[idx];
        if (idx-fullOffset>=2) return pp[idx-icage*2];
        return Complex{ 0, 0 };
    };

    auto PPDiff = [PPat](const Complex *pp, int i, int j) { return PPat(pp,i)-PPat(pp,j); };

#else
    const int next = (i < n - 1) ? i + 1 : 0;
    const int prev = (i > 0) ? i - 1 : n - 1;
    auto PPDiff = [](const Complex *pp, int i, int j) { return pp[i]-pp[j]; };
#endif


    const Complex vn_minus_v = v[next]-v[i];
    const Complex vp_minus_v = v[prev]-v[i];

    Complex s1 = PPDiff(phi,next, i) / vn_minus_v;
    Complex s2 = PPDiff(phi,prev, i) / vp_minus_v;
    dSimlarity[i] = abs(s1 - s2);

    s1 += PPDiff(dphi,next,i)**t / vn_minus_v;
    s2 += PPDiff(dphi,prev,i)**t / vp_minus_v;
    dSimlarity[i+n*2] = abs(s1 - s2);


    const Complex* psy = phi + npp;
    const Complex* dpsy = dphi + npp;

    s1 = PPDiff(psy,next,i) / vn_minus_v;
    s2 = PPDiff(psy,prev,i) / vp_minus_v;
    dSimlarity[i+n] = abs(s1 - s2);

    s1 += PPDiff(dpsy,next,i)**t / vn_minus_v;
    s2 += PPDiff(dpsy,prev,i)**t / vp_minus_v;
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

template<class Complex, class real>
__global__ void buildP2PMatrixQ(const Complex *C, int m, int n, real *M)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= m || j >= n) return;

    const Complex &z = C[i + j*m];

    const int m2 = m * 2;
    M[i +  j*m2]      =  z.x;
    M[i + (j+n)*m2]   =  z.x;
    M[i + (j+n*2)*m2] = -z.y;
    M[i + (j+n*3)*m2] = -z.y;

    M[i+m +  j*m2]      =   z.y;
    M[i+m + (j+n)*m2]   =  -z.y;
    M[i+m + (j+n*2)*m2] =   z.x;
    M[i+m + (j+n*3)*m2] =  -z.x;
}

//////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ double isometric_energy_from_fz2gz2(double fz2, double gz2, int energy_type, double s=1.)
{
    double e;
    switch (energy_type){
    case ISO_ENERGY_SYMMETRIC_DIRICHLET:
        e = (fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2));   // TODO, shift energy to 0, to see if it make any difference numerically
        return e;
    case ISO_ENERGY_EXP_SD:
        //e = fabs((fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2))-2);   // TODO, shift energy to 0
        e = (fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2));
        return exp(s*e);
    case ISO_ENERGY_AMIPS:
        //return exp(abs((2 * s*(fz2 + gz2) + 1) / (fz2 - gz2) + (fz2 - gz2)) - 2 - 2 * s);  // shift energy to 0
        return exp((2 * s*(fz2 + gz2) + 1) / (fz2 - gz2) + (fz2 - gz2)); 
    default:
        assert(false); // not implemented
    }

    return 0;  
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void isometry_energy(real *en, const Complex *fzgz, const Complex *dfzgz, const real *t, int m, int energy_type, double epow)
{
	const int it = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0;

    const int ien = blockIdx.y;
#pragma unroll
    for (int j = 0; j < nItemPerReduceThread; j++) {
        int i = it*nItemPerReduceThread + j;
        if (i >= m) continue;
        double fz2 = dfzgz?abs2(fzgz[i] + dfzgz[i] * t[ien]):abs2(fzgz[i]);
        double gz2 = dfzgz?abs2(fzgz[i + m] + dfzgz[i + m] * t[ien]):abs2(fzgz[i+m]);
        e += isometric_energy_from_fz2gz2(fz2, gz2, energy_type, epow);
    }

    using BlockReduceT = cub::BlockReduce<double, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
    __shared__ typename BlockReduceT::TempStorage  temp_storage;

    real block_en = real( BlockReduceT(temp_storage).Sum(e) );

    if (threadIdx.x == 0) 
        atomicAdd(en+ien, block_en);
}



//////////////////////////////////////////////////////////////////////////
// for Newton reducing redundant variables 
// compute dpp = N*( (N'*HF*N)\(N'*fC2Rv(-g)), HF is full hessian, dpp = N*x, x are free (independent) variables
// compute N'*g for particular N for harmonic map on multiply connected domain
template<class Complex, class R>
__global__ void computeReducedGradient(const Complex *Gin, R *Gout)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    int nCage = cageOffsets_g.n;
    int npp = cageOffsets_g.offsets[nCage] + nCage - 1;
    int outCageSz = cageOffsets_g.offsets[1];
    int nvar = npp * 2;
    int nFreeVar = nvar - nCage;

    if (i >= nFreeVar) return;

    const int idLastOutCagePsy = npp + outCageSz - 1;
    int xskip = (i < idLastOutCagePsy) ? 0 : 1;

    Complex v = Gin[i+xskip];

    bool mergeVal = ( i >= npp - nCage + 1 && i < npp );
    if (mergeVal) 
        v += conj(Gin[i+xskip+npp]);

    Gout[i] = v.x;
    Gout[i+nFreeVar] = v.y;
}


//////////////////////////////////////////////////////////////////////////
// compute N'*HF*N 
template<class R>
__global__ void computeReducedHessian(const R *Hin, R *Hout)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    int nCage = cageOffsets_g.n;
    int npp = cageOffsets_g.offsets[nCage] + nCage - 1;
    int outCageSz = cageOffsets_g.offsets[1];
    int nvar = npp * 2;
    int nFreeVar = nvar - nCage;

    int n2 = nFreeVar * 2;
    if (i >= n2 || j >= n2) return;

    const int idLastOutCagePsy = npp + outCageSz - 1;
    int xskip = (i < idLastOutCagePsy) ? 0 : (i<nFreeVar+idLastOutCagePsy?1:2);
    int yskip = (j < idLastOutCagePsy) ? 0 : (j<nFreeVar+idLastOutCagePsy?1:2);

    int nImSkip = nCage - 1;
    xskip += (i < nFreeVar) ? 0 : nImSkip;
    yskip += (j < nFreeVar) ? 0 : nImSkip;

    const int baseIdx = (i + xskip) + (j + yskip)*nvar * 2;
    R v = Hin[baseIdx];

    auto fMergeSign = [nFreeVar](int i) {return i < nFreeVar ? 1 : -1; };
    int iRe = i<nFreeVar?i:i-nFreeVar;
    bool mergeRow = ( iRe >= npp - nCage + 1 && iRe < npp );
    if (mergeRow) 
        v += Hin[baseIdx+npp]*fMergeSign(i);

    int jRe = j<nFreeVar?j:j-nFreeVar;
    bool mergeColumn = (jRe >= npp - nCage + 1 && jRe < npp);
    if (mergeColumn)
        v += Hin[baseIdx+npp*nvar*2]*fMergeSign(j);

    if(mergeColumn&&mergeRow)
        v += Hin[baseIdx+npp*nvar*2 + npp]*(fMergeSign(i)*fMergeSign(j));

    Hout[i + j*n2] = v;
}

//////////////////////////////////////////////////////////////////////////
// compute dpp = N'*x
template<class Complex, class R>
__global__ void computeFullDPP(const R *dpp_in, Complex *dpp_out, R scale=1)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    int nCage = cageOffsets_g.n;
    int npp = cageOffsets_g.offsets[nCage] + nCage - 1;
    int outCageSz = cageOffsets_g.offsets[1];
    int nvar = npp * 2;
    int nFreeVar = nvar - nCage;

    if (i >= nvar) return;

    Complex v = { 0, 0 };
    const int idLastOutCagePsy = npp + outCageSz - 1;
    //if (i == idLastOutCagePsy) v = { 0, 0 };

    if (i != idLastOutCagePsy) {
        int xskip = (i < idLastOutCagePsy) ? 0 : (i<npp+cageOffsets_g.offsets[nCage]?-1:-npp);
        char imsign = (xskip < -1)?-1:1;

        v = { dpp_in[i + xskip], dpp_in[i + xskip + nFreeVar]*imsign };
    }

    dpp_out[i] = v*scale;
}



//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_map_sum_dargfz(real *sum, const Complex *fz, const Complex *dfz, const int *nextSampleInSameCage, int m, const real *t)
{
    const int it = blockIdx.x*blockDim.x + threadIdx.x;
    const int isum = blockIdx.y;

    double s = 0;

#pragma unroll
    for (int j = 0; j < nItemPerReduceThread; j++) {
        int i = it*nItemPerReduceThread + j;
        if (i >= m) continue;

        const int k = nextSampleInSameCage ? nextSampleInSameCage[i] : ((i < m - 1) ? i + 1 : 0);
        const Complex fz0 = fz[i] + dfz[i] * t[isum];
              Complex fz1 = fz[k] + dfz[k] * t[isum];

        fz1 = fz1*conj(fz0);
        double dtheta = atan2(fz1.y, fz1.x);
        s += dtheta;
    }

    using BlockReduceT = cub::BlockReduce<double, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
    __shared__ typename BlockReduceT::TempStorage  temp_storage;

    real block_sum = real(BlockReduceT(temp_storage).Sum(s));

    if (threadIdx.x == 0) 
        atomicAdd(sum + isum,   block_sum);
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_map_distortion_bounds(real *bounds, const Complex *fzgz, const Complex *dfzgz, int m,
    const Complex *fzzgzz, const Complex *dfzzgzz, const real* delta_L_fzgz, 
    const real* sampleSpacings, const int *nextSampleInSameCage, const real *t, const real step0)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    double all_bounds[6] = { 0, 0, 0, 0, 0, 0 };
    const int istep = blockIdx.y;

    if (i < m) {
        const real tt = t[istep];
        const int k = nextSampleInSameCage ? nextSampleInSameCage[i] : ((i < m - 1) ? i + 1 : 0);
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

       if(i<5)  __syncthreads();
    }
}


//////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void alpha_beta_from_fz2gz2(double fz2, double gz2, double alphas[2], double* betas, int energy_type, double s=1.)
{
    switch (energy_type){
    case ISO_ENERGY_SYMMETRIC_DIRICHLET: {
        // fzgzb(:, 1).*(1 - ((fzgzb2*[1;-1]).^-3).*(fzgzb2*[1;3])) 
        // fzgzb(:, 2).*(1 + ((fzgzb2*[1;-1]).^-3).*(fzgzb2*[3;1]))
        double e = isometric_energy_from_fz2gz2(fz2, gz2, ISO_ENERGY_SYMMETRIC_DIRICHLET, s);
        double w = 1;
        alphas[0] = w*(1 - (fz2 + 3 * gz2) / pow3(fz2 - gz2));
        alphas[1] = w*(1 + (3 * fz2 + gz2) / pow3(fz2 - gz2));

        if (betas) {
            double c = 2 / pow4(fz2 - gz2);
            betas[0] = c * (fz2 + 5 * gz2);
            betas[1] = c * (5 * fz2 + gz2);
            betas[2] = -3 * c * (fz2 + gz2);
            if (s != 1) {
                double d = (s - 1) / s / e;
                betas[0] = betas[0] * w + sqr(alphas[0]) *d;
                betas[1] = betas[1] * w + sqr(alphas[1]) *d;
                betas[2] = betas[2] * w + alphas[0] * alphas[1] * d;
            }
        }
        break;
    }
    case ISO_ENERGY_EXP_SD: {
        double e = isometric_energy_from_fz2gz2(fz2, gz2, ISO_ENERGY_EXP_SD, s);
        alphas[0] = s*e*(1 - (fz2 + 3 * gz2) / pow3(fz2 - gz2));
        alphas[1] = s*e*(1 + (3 * fz2 + gz2) / pow3(fz2 - gz2));

        if (betas) {
            double c = 2 / pow4(fz2 - gz2);
            betas[0] = c * (fz2 + 5 * gz2) * s* e + sqr(alphas[0]) / e;
            betas[1] = c * (5 * fz2 + gz2) * s* e + sqr(alphas[1]) / e;
            betas[2] = -3 * c * (fz2 + gz2) * s* e + alphas[0] * alphas[1] / e;
        }
        break;
    }
    case ISO_ENERGY_AMIPS: {
        double e = isometric_energy_from_fz2gz2(fz2, gz2, ISO_ENERGY_AMIPS, s);
        alphas[0] = e*(1 - (4 * s*gz2 + 1) / sqr(fz2 - gz2));
        alphas[1] = e*(-1 + (4 * s*fz2 + 1) / sqr(fz2 - gz2));

        if (betas) {
            double c = 2 / pow3(fz2 - gz2);
            betas[0] = c*(4 * s*gz2 + 1);
            betas[1] = c*(4 * s*fz2 + 1);
            betas[2] = -c*(2 * s*(fz2 + gz2) + 1);

            betas[0] = betas[0] * e + sqr(alphas[0]) / e;
            betas[1] = betas[1] * e + sqr(alphas[1]) / e;
            betas[2] = betas[2] * e + alphas[0] * alphas[1] / e;
        }
        break;
    }
    default:
        assert(false); // energy type not implemented
    }
}


//////////////////////////////////////////////////////////////////////////
template<class Complex>
__global__ void isometry_gradient_diagscales(Complex *x, const Complex *fzgz, int m, int energy_type, double epow) 
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= m) return;

    using real = decltype(fzgz[i].x);
	const int j = i + m;

    double alphas[2];
    alpha_beta_from_fz2gz2(abs2(fzgz[i]), abs2(fzgz[i+m]), alphas, nullptr, energy_type, epow);
	x[i] = fzgz[i] * real( alphas[0] );
	x[j] = fzgz[j] * real( alphas[1] );
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void isometry_hessian_diagscales(real *const s, const Complex *fzs, const Complex *gzs, const int *samples, int m, int energy_type, real epow, bool spdHessian = true)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m) return;

    const int sidx = samples ? samples[i] : i;
    const Complex &fz = fzs[sidx], &gz = gzs[sidx];
    const double fz2 = abs2(fz), gz2 = abs2(gz);

    double alphas[2], betas[3];
    alpha_beta_from_fz2gz2(fz2, gz2, alphas, betas, energy_type, epow);

    if (spdHessian) {
        if (SPD_hessian_modification_is_simple[energy_type]) {
            if (alphas[0] < 0) {
                betas[0] += alphas[0] / 2 / fz2;
                alphas[0] = 0;
            }
        }
        else if(gz2>1e-15){ // general SPD modification, take care of identity map, where |gz|=0 causes problem for the following modification
            double t1 = alphas[0] + 2 * betas[0] * fz2;
            double t2 = alphas[1] + 2 * betas[1] * gz2;
            const double s1 = t1 + t2, s2 = t1 - t2;
            t1 = sqrt(sqr(s2) + 16 * sqr(betas[2])*fz2*gz2);
            double lambda3 = s1 + t1, lambda4 = s1 - t1;

            t1 = (lambda3 - 2 * alphas[0] - 4 * betas[0] * fz2) / (4 * betas[2] * gz2);
            t2 = (lambda4 - 2 * alphas[0] - 4 * betas[0] * fz2) / (4 * betas[2] * gz2);

            lambda3 = max(lambda3, 0.) / (fz2 + gz2*sqr(t1));
            lambda4 = max(lambda4, 0.) / (fz2 + gz2*sqr(t2));

            alphas[0] = max(alphas[0], 0.);
            alphas[1] = max(alphas[1], 0.);

            betas[0] = lambda3 + lambda4 - alphas[0] / 2 / fz2;
            betas[1] = lambda3*sqr(t1) + lambda4*sqr(t2) - alphas[1] / 2 / gz2;
            betas[2] = lambda3*t1 + lambda4*t2;
        }
    }

    alphas[0] *= 2;  alphas[1] *= 2;
    betas[0] *= 4;  betas[1] *= 4; betas[2] *= 4;

    real ss3     = real(fz.x * fz.y*betas[0]);
    s[i]         = real(sqr(fz.x)*betas[0] + alphas[0]);
    s[i + m]     = ss3;
    s[i + m * 2] = ss3;
    s[i + m * 3] = real(sqr(fz.y)*betas[0] + alphas[0]);
                                             
    i += m * 4;
    ss3          = real(gz.x * gz.y * betas[1]);
    s[i]         = real(sqr(gz.x)*betas[1] + alphas[1]);
    s[i + m]     = ss3;
    s[i + m * 2] = ss3;
    s[i + m * 3] = real(sqr(gz.y)*betas[1] + alphas[1]);

    i += m * 4;
    s[i]         = real(fz.x * gz.x * betas[2]);
    s[i + m]     = real(fz.x * gz.y * betas[2]);
    s[i + m * 2] = real(fz.y * gz.x * betas[2]);
    s[i + m * 3] = real(fz.y * gz.y * betas[2]);
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
    const real *steps, real *en, int isoEnergyType, real epow, real lambda, int nsteps = 1, real *en_isometry = nullptr)
{
    isometry_energy <<<dim3(reduceBlockNum(m), nsteps), threadsPerBlock >>> (en, fzgz, dfzgz, steps, m, isoEnergyType, epow);
    CUDA_CHECK_ERROR;

    print_gpu_value(en, "isometric energy", false);
    
    // save isometric energies separately, for potential display/debug later 
    if(en_isometry) cudaMemcpyAsync(en_isometry, en, sizeof(real)*nsteps, cudaMemcpyDeviceToDevice);

    // add P2P energy
    p2p_energy << <dim3(blockNum(nP2P), nsteps), threadsPerBlock >> > (en, fgP2P, dfgP2P, steps, bP2P, nP2P, lambda);
    CUDA_CHECK_ERROR;

    print_gpu_value(en, "total energy");
};

//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_line_search(const Complex *fzgz, const Complex *dfzgz, const Complex *fgP2P, const Complex *dfgP2P, const int *nextSampleInSameCage, const Complex* bP2P, int m, int nP2P,
    real *steps, real *e, int isoEnergyType, real isoepow, const real *dgdotdpp_and_norm2dpp, real lambda, int enEvalsPerKernel)
{
    const double ls_alpha = 0.2;
    const double ls_beta = 0.5;

    const real dgdotfz = *dgdotdpp_and_norm2dpp;
    const real normdpp = sqrt(dgdotdpp_and_norm2dpp[1]);

    //real en[1];
    real *const en = e + 1;
    real *const sum_dargfz = en + enEvalsPerKernel*2;
    auto fQPEstim = [&](real t) { return *e + ls_alpha*t*dgdotfz; };

    // use only one thread for linesearch
    if (threadIdx.x > 0) return;

    bool done = false;
    while (!done){
        for (int i = 1; i < enEvalsPerKernel; i++) steps[i] = *steps / (1 << i); // powf(ls_beta, i);

        //////////////////////////////////////////////////////////////////////////
        //for (int i = 0; i < enEvalsPerKernel; i++) en[i] = 0;
        cudaMemsetAsync(en, 0, sizeof(real)*enEvalsPerKernel);
#ifdef _DEBUG
        p2p_harmonic_full_energy(fzgz, dfzgz, fgP2P, dfgP2P, bP2P, m, nP2P, steps, en, isoEnergyType, isoepow, lambda, enEvalsPerKernel, en + enEvalsPerKernel);
        cudaDeviceSynchronize();
        for (int i = 0; i < enEvalsPerKernel; i++) {
            gpu_print(en[i + enEvalsPerKernel], "isometric energy", false);
            gpu_print(en[i], "total energy");
        }
#else
        p2p_harmonic_full_energy(fzgz, dfzgz, fgP2P, dfgP2P, bP2P, m, nP2P, steps, en, isoEnergyType, isoepow, lambda, enEvalsPerKernel);
#endif

        //////////////////////////////////////////////////////////////////////////
        // argument principal to make sure fz does not vanish inside the domain, works if the Lipschitz of fz can certify that |fz|_min>0 for all boundary segments 
        cudaMemsetAsync(sum_dargfz, 0, sizeof(real)*enEvalsPerKernel);
        harmonic_map_sum_dargfz <<<dim3(reduceBlockNum(m), enEvalsPerKernel), threadsPerBlock >>> (sum_dargfz, fzgz, dfzgz, nextSampleInSameCage, m, steps);
        CUDA_CHECK_ERROR;
        cudaDeviceSynchronize();

        for (int i = 0; i < enEvalsPerKernel; i++) {
            real ls_t = steps[i];
            const real argment_principal_eps = 1;
            gpu_print(sum_dargfz[i], "argument principal");
            if (ls_t*normdpp < minDeformStepNorm || (en[i] < fQPEstim(ls_t) && fabs(sum_dargfz[i]) < argment_principal_eps)) {
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
}


//////////////////////////////////////////////////////////////////////////
template<class Complex, class real>
__global__ void harmonic_map_validate_bounds(const Complex *fzgz, const Complex *dfzgz, int m, const Complex *fzzgzz, const Complex *dfzzgzz, 
    const real *delta_L_fzgz, const real* sampleSpacings, const int *nextSampleInSameCage, real *ls_t, real *bounds, const real *norm2dpp, int enEvalsPerKernel)
{
    const double ls_beta = 0.5;
    const real normdpp = sqrt(*norm2dpp);

    const real step0 = *ls_t;

    // need at least one run for the distortion bounds, TODO: or just report that it has converged
    do{        
        cudaMemsetAsync(bounds, 0, sizeof(real)*6);
        harmonic_map_distortion_bounds <<<blockNum(m), threadsPerBlock >>> (bounds, fzgz, dfzgz, m, fzzgzz, dfzzgzz, delta_L_fzgz, sampleSpacings, nextSampleInSameCage, ls_t, step0);

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
	vecC g;    // gradient  // isometry or p2p or combined, size 2n+1
	vecR gR;   // gradient in real
	int m;	   // number of sample, #row in D2
	int n;	   // number of cage vertex, #column in D2
    int mh;    // number of sample for hessian, can be <= m, the first mh rows of D2 is used for hessian computation

    std::vector<int> cageVIdxOffsets; // (begin) vertex index offsets of the out cage and interior holes (except first 2 vertices), correspond to phi/psi

    int isoetype = ISO_ENERGY_SYMMETRIC_DIRICHLET;
    real isoepow = 1.; // raise isometry energy to power

	vecC fzgz;
	vecR DRDI;

    cublasHandle_t cub_hdl = 0;
	vecC constants;
    Complex *pOne = nullptr, *pZero = nullptr, *pMinusOne = nullptr;
    Complex *p2Lambda = nullptr, *pHessianSampleRate = nullptr;        // constants 


    // whether/how to modify the hessian of the isometry energy to be SPD
    // 0: no modification vanilla hessian, 1: per-sample SPD hessian, 2: SPD hessian using eigenfactorization on hessian of the full mapping
    enum { ISO_HESSIAN_VANILLA = 0, ISO_HESSIAN_SPD_PERSAMPLE = 1, ISO_HESSIAN_SPD_FULLEIG=2 };
    int modifyHessianSPD = ISO_HESSIAN_SPD_PERSAMPLE; 


    vecR h_tmpstorage; // for computing hessian
    vecR h_diagscales; // for computing hessian / gradient



    const Complex *C2 = nullptr;      // Cauchy coord at p2p 
    int nP2P = 0;
    vecC  fgP2P;
    vecC phipsy_tmpstorage, bP2P_tmpstorage, matGradP2P;
    real P2P_weight = 0;

    vecR C2C2BarReal;

    cuCholSolverDn<real> solverCHOL;
    cuLUSolverDn<real> solverLU;

    int linearSolvePref = LINEAR_SOLVE_PREFER_CHOLESKY;
    real deltaFixSPDH = real(1e-15);

    static const bool runDoublePrecision = std::is_same<real, double>::value;


	cuHarmonicMap(const Complex *d_D2, int nSample, int dim, const int *hessian_sample_indices, int nHessSample, const std::vector<int> &cageVertexOffsets) 
        :D2(d_D2), m(nSample), n(dim), hessian_samples(hessian_sample_indices), mh(nHessSample), phipsy(dim*2), 
        g(dim*2+1), gR(dim*4), h(dim*dim*16), cageVIdxOffsets(cageVertexOffsets)
    {
        myZeroFill(g.data() + n * 2);
        if (cageVIdxOffsets.empty()) cageVIdxOffsets.insert(cageVIdxOffsets.end(), { 0, n });
    }

    ~cuHarmonicMap() { cublasDestroy(cub_hdl); }

    void init();

    void setupP2P(const Complex *d_C2, int numberP2P, real lambda);

    int nCages() const { return cageVIdxOffsets.size() - 1; }
    int nAllVars() const { return cageVIdxOffsets.back()+(nCages()-1)*3; }     // phi_c0, phi_c1, ... w1, w2, ...
    int numFreeVars() const { return n * 2 - nCages(); }

    void update_bP2P(const Complex *bP2P) {   // bP2P: target positions of p2p constraints, update matGradP2P (last column) for p2p gradient evaluation
        ChooseCublasFunc(axpy, Daxpy, Saxpy);
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
		isometry_gradient_diagscales<<<blockNum(m), threadsPerBlock>>>(gradscales, fzgz.data(), m, isoetype, isoepow);
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

        hess = hess ? hess : h.data();
        const int n2 = n * 2;

        //////////////////////////////////////////////////////////////////////////
        // add p2p grad
        const real *d_CCBar = C2C2BarReal.data();        // [C2 conj(C2) -bP2P]
        // compensate for sparse sampling for the hessian
        cublasStatus_t sta = gemm(cub_hdl, CUBLAS_OP_T, CUBLAS_OP_N, n2*2, n2*2, nP2P*2, (real*)p2Lambda, d_CCBar, nP2P*2, d_CCBar, nP2P*2, (real*)pHessianSampleRate, hess, n2*2);
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
    buildP2PMatrixQ<Complex,real> <<<dim3(ceildiv(nP2P, 4), ceildiv(n, 32)), dim3(4, 32) >>> (C2, nP2P, n, C2C2BarReal.data());
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

    //////////////////////////////////////////////////////////////////////////
    // copy cage offsets to gpu constant memory
    CageVertexIndexOffsets cageoffs;
    cageoffs.n = cageVIdxOffsets.size()-1;
    assert(cageVIdxOffsets.size() <= CageVertexIndexOffsets::maxHarmonicMapCageNumber);
    std::copy(cageVIdxOffsets.cbegin(), cageVIdxOffsets.cend(), cageoffs.offsets);

    cudaMemcpyToSymbolAsync(cageOffsets_g, &cageoffs, sizeof(CageVertexIndexOffsets));
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

        // reduce matrix A loading time TODO: make sure d_Htmp2 has enough memory >2n*4n!
        sta = rgemm(cub_hdl, CUBLAS_OP_T, CUBLAS_OP_N, n2, n4, mh, pOneR, d_DRDI, mh2, d_Htmp1, mh, pZeroR, d_Htmp2, n2);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, n, n2, pOneR, d_Htmp2, n4, pMinusOneR, d_Htmp2 + n*3, n4, subH, n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
        sta = geam(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, n, n2, pOneR, d_Htmp2 + n, n4, pOneR, d_Htmp2 + n*2, n4, subH + n, n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
    };


    real *const diagscales = h_diagscales.data();
    hess = hess ? hess : h.data();

    isometry_hessian_diagscales <<<blockNum(mh), threadsPerBlock >>> (diagscales, fzgz.data(), fzgz.data()+m, hessian_samples, mh, isoetype, isoepow, modifyHessianSPD==ISO_HESSIAN_SPD_PERSAMPLE);
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

        // step 2: eig
        syevd(hdl, jobz, uplo, n4, hess, n4, D.data(), buffer.data(), bufferSize, info.data());

        // step 3: take sqrt of D: eigen values
        sqrt_non_negative_clamp<real> <<<blockNum(n4), threadsPerBlock>> > (D.data(), D.data(), n4);

        cuVector<real> sqrtHess(n4*n4);
        sta = dgmm(cub_hdl, CUBLAS_SIDE_RIGHT, n4, n4, hess, n4, D.data(), 1, sqrtHess.data(), n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        sta = rgemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_T, n4, n4, n4, pOneR, sqrtHess.data(), n4, sqrtHess.data(), n4, pZeroR, hess, n4);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
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
    
    // prepare for Newton
    conjugate_inplace <<<blockNum(n), threadsPerBlock >>> (grad + n, n);
    CUDA_CHECK_ERROR;

    // after conjugate, so that it's consistent for compute dg_dot_dpp later
    myCopy_n(grad, n2, grad_out);

    recordEvent(times[0]); // record gradient time
    
    real* const hess = h.data();
    hessian_full(hess);

    recordEvent(times[1]); // record hessian time

    //////////////////////////////////////////////////////////////////////////
    real *const gradR = gR.data();

    cuSolverDN<real> *solver = &solverLU;
    if (linearSolvePref == LINEAR_SOLVE_FORCE_CHOLESKY || (SPDHessian != ISO_HESSIAN_VANILLA&&linearSolvePref == LINEAR_SOLVE_PREFER_CHOLESKY)) solver = &solverLU;

    const int nFreeVar = numFreeVars(); // n*2-nCages(), number of complex variables to solve
    computeReducedGradient <<<blockNum(nFreeVar), threadsPerBlock >>> (grad, gradR);
    solver->init(nFreeVar*2); // allocate memory first;
    computeReducedHessian <<<dim3(ceildiv(nFreeVar*2,32), ceildiv(nFreeVar*2, 32)), dim3(32, 32)>>>(hess, solver->A.data());

    int solver_stat = solver->factor();
    assert(solver_stat == 0);
    solver->solve(gradR);

    computeFullDPP <<<blockNum(n * 2), threadsPerBlock >>> (gradR, dpp, real(-1));

    recordEvent(times[2]); // record solve time
}


#undef ChooseCusolverDnFunc
#undef ChooseCublasFunc
