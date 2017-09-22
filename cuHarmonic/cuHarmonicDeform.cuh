#include "harmonic_map.cuh"

enum OptimMethod { OM_GRADIENT_DESCENT = 0, OM_NEWTON, OM_NEWTON_SPDH, OM_NEWTON_SPDH_FULLEIG, NUM_OPTIM_METHOD };
const char* OptimizationMethodNames[] = {"Gradient Descent", "Newton", "Newton SPDH", "Newton Eigen" };

template<class R>
__global__ void stepNorm_from_stepSize_sqrNorm(const R* stepSize, const R* sqrStepNorm0, R* stepnorm) { *stepnorm = *stepSize*sqrt(*sqrStepNorm0); }


template<class Complex, class real>
struct HarmonicMapValidationInput
{
    const Complex* v = nullptr;   // cage: virtual vertices
    const Complex* E2 = nullptr;  // second order derivatives of Cauchy coordinates
    const real* L = nullptr;      // Lipschitz constants for computing fz, gz
    const int* nextSampleInSameCage = nullptr;

    HarmonicMapValidationInput(const Complex *pv=nullptr, const Complex *pE2=nullptr, const real* pL=nullptr, const int* pNextSample=nullptr)
        :v(pv), E2(pE2), L(pL), nextSampleInSameCage(pNextSample) {}
};

// todo: sampleSpacings may be used to improve the energy/gradient/hessian integral

template<class Complex, class real>
std::vector<double> cuHarmonicDeform(const Complex *d_D2, const Complex *d_C2, const Complex *d_bP2P, Complex *d_phipsyIters, const int *hessian_samples,
    int m, int mh, int n, const std::vector<int> &cageOffsets, int nP2P, int isoEnergyType, real isoepow, real lambda, int nIter, 
    HarmonicMapValidationInput<Complex, real> validationData,
    const real* sampleSpacings = nullptr, int enEvalsPerKernel = 8, int optimization_method = OM_NEWTON_SPDH, 
    int reportIterationStats = 1, int linearSolvePref = LINEAR_SOLVE_PREFER_CHOLESKY, real deltaFixSPDH = 1e-15)
{
    //////////////////////////////////////////////////////////////////////////
    using gpuVecComplex = cuVector<Complex>;
    using gpuVecReal = cuVector<real>;

    const bool runDoublePrecision = std::is_same<decltype(d_D2->x), double>::value;

    printf("Config: %s, %s, %.1f%%HS, wtP2P: %.1e, PDfix=%.1e\n",
        OptimizationMethodNames[optimization_method], IsometricEnergyNames[isoEnergyType], mh*100. / m, lambda, deltaFixSPDH);

    using gemmtype = std::conditional_t<runDoublePrecision, decltype(&cublasZgemm), decltype(&cublasCgemm)>;
    using rgemmtype = std::conditional_t<runDoublePrecision, decltype(&cublasDgemm), decltype(&cublasSgemm)>;
    using axpytype = std::conditional_t<runDoublePrecision, decltype(&cublasDaxpy), decltype(&cublasSaxpy)>;
    using rdottype = std::conditional_t<runDoublePrecision, decltype(&cublasDdot), decltype(&cublasSdot)>;
    using scaltype = std::conditional_t<runDoublePrecision, decltype(&cublasDscal), decltype(&cublasSscal)>;

    const auto gemm = runDoublePrecision?(gemmtype)&cublasZgemm:(gemmtype)&cublasCgemm;
    const auto rgemm = runDoublePrecision?(rgemmtype)&cublasDgemm:(rgemmtype)&cublasSgemm;
    const auto axpy = runDoublePrecision?(axpytype)&cublasDaxpy:(axpytype)&cublasSaxpy;
    const auto rdot = runDoublePrecision?(rdottype)&cublasDdot:(rdottype)&cublasSdot;
    const auto scal = runDoublePrecision?(scaltype)&cublasDscal:(scaltype)&cublasSscal;

    const int n2 = n*2;

    cuHarmonicMap<Complex, real> HM(d_D2, m, n, hessian_samples, mh, cageOffsets);
    HM.linearSolvePref = linearSolvePref;
    HM.deltaFixSPDH = deltaFixSPDH;
    HM.init();
    HM.isoetype = isoEnergyType;
    HM.isoepow = isoepow;
    HM.setupP2P(d_C2, nP2P, lambda);
    HM.update_bP2P(d_bP2P);


    Complex *const d_fzgz0 = HM.fzgz.data();
    Complex *const d_fgP2P = HM.fgP2P.data();
    Complex *const d_phipsy= HM.phipsy.data();


    //////////////////////////////////////////////////////////////////////////
    cublasHandle_t cub_hdl = HM.cub_hdl;

    const Complex myConsts_cpu[] = { { 1, 0 }, { -1, 0 }, { 0, 0 } };
    gpuVecComplex myConts(std::end(myConsts_cpu)-std::begin(myConsts_cpu), myConsts_cpu);

    const Complex *pOne = myConts.data();
    const Complex *pMinusOne = pOne+1, *pZero = pOne+2;

    HM.update_phipsy(d_phipsyIters);
    HM.update_fzgz(); 
    HM.update_fgP2P();

    auto fMyEnergy = [d_fzgz0, d_fgP2P, lambda, d_bP2P, m, nP2P, isoEnergyType, isoepow](const Complex *dfzgz, real *t, const Complex *dfgP2P, real *en) { 
        p2p_harmonic_full_energy(d_fzgz0, dfzgz, d_fgP2P, dfgP2P, d_bP2P, m, nP2P, t, en, isoEnergyType, isoepow, lambda, 1);
    };

    auto fMyGrad = [&HM, n](Complex *const grad){
        HM.isometry_gradient(grad);

        conjugate_inplace<<<blockNum(n), threadsPerBlock>>>(grad + n, n);
		CUDA_CHECK_ERROR;

        HM.p2p_gradient(grad);
    };

    gpuVecComplex grad(n2);  
    Complex *d_grad = grad.data();

    gpuVecComplex dpp(n2);                              // [dphi dpsy]
    Complex *d_dpp = dpp.data();

    gpuVecComplex dfzgz(m*2); 
    Complex *const d_dfzgz = dfzgz.data();

    gpuVecComplex dfgP2P(nP2P*2); 
    Complex *const d_dfgP2P = dfgP2P.data();

    gpuVecReal steps_ls = std::vector<real>(nIter*enEvalsPerKernel, 1.); // setp sizes for line search, max step: 1, 
    gpuVecReal dgdotdpp(nIter*2);  dgdotdpp.zero_fill();

    //////////////////////////////////////////////////////////////////////////
    // following array contains: [e0: initial/current, i.e. before linesearch; total en in linesearch; isometric en in linesearch; argment principal approximation; approximate check for (34) in Chen15
    gpuVecReal energies(enEvalsPerKernel * 4 + 1);  energies.zero_fill(); 
    real *const d_en = energies.data();
    fMyEnergy(nullptr, nullptr, nullptr, d_en);

    auto fp2pEn = [d_fgP2P, d_bP2P, nP2P, d_en]() {
        myZeroFill(d_en);
        p2p_energy<Complex, real><<<blockNum(nP2P), threadsPerBlock>>> (d_en, d_fgP2P, nullptr, nullptr, d_bP2P, nP2P, 1.0);
        CUDA_CHECK_ERROR;
        return copyValFromGPU(d_en);
    };
    
    cublasStatus_t sta;

    //////////////////////////////////////////////////////////////////////////
    gpuVecComplex fzzgzz;
    gpuVecReal delta_L_fzgz;
    gpuVecReal distortion_bounds;
    gpuVecReal abs_dS0dSt0;
    const int nAllVars = HM.nAllVars();

    const bool needValidateMap = (validationData.v && validationData.E2 && validationData.L);
    if (needValidateMap) {
        // fzz, gzz, dfzz, dgzz
        fzzgzz.resize(m * 4);

        // delta_L_fz
        delta_L_fzgz.resize(m * 4);

        // sigma1, sigma2, k, global/samples
        distortion_bounds.resize(nIter*6);

        abs_dS0dSt0.resize(nAllVars * 4);

        sta = gemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, m, 2, n, pOne, validationData.E2, m, d_phipsy, n, pZero, fzzgzz.data(), m);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
    }

    enum { TIME_BEGIN = 0, TIME_GRAD, TIME_HESSIAN, TIME_SOLVE, TIME_LS, TIME_END, TIME_NUMS };

    std::vector<cudaEvent_t> times(nIter * TIME_NUMS);
    gpuVecReal stats_energies_steps( (nIter+1) * 3);  // step norm, isometric energy, full energy
    if (reportIterationStats) {
        for (auto &tt : times)cudaEventCreate(&tt);

        stats_energies_steps.zero_fill();
        p2p_energy<Complex, real> <<<blockNum(nP2P), threadsPerBlock >>> (stats_energies_steps.data() + 1, d_fgP2P, nullptr, nullptr, d_bP2P, nP2P, lambda);
        myCopy_n(d_en, 1, stats_energies_steps.data() + 2);
    }
    auto recordEvents = [reportIterationStats](cudaEvent_t *t, int nEvts) { if (reportIterationStats) for(int i=0; i<nEvts; i++) cudaEventRecord(t[i], 0); };
    
    for (int it = 0; it < nIter; it++)  {
        real *step = steps_ls.data() + it*enEvalsPerKernel;  // *cur_ls_t == 1.
        real *const dgdotdpp_and_norm2dpp = dgdotdpp.data() + it*2;

        cudaEvent_t *curIterTimes = &times[it*TIME_NUMS];
        recordEvents(curIterTimes + TIME_BEGIN, 1);

        // find step to reduce energy
        switch (optimization_method) {
        case OM_GRADIENT_DESCENT:
            fMyGrad(d_grad);
            
            recordEvents(curIterTimes+TIME_GRAD, 3); // grad, no hessian, no solve 

            d_dpp = d_grad;         // gradient descending
            sta = scal(cub_hdl, n2 * 2, (real*)pMinusOne, (real*)d_dpp, 1);
            ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
            break;
        case OM_NEWTON:
        case OM_NEWTON_SPDH:
        case OM_NEWTON_SPDH_FULLEIG:
            HM.computeNewtonStep(d_dpp, d_grad, optimization_method - OM_NEWTON, reportIterationStats ? (curIterTimes + TIME_GRAD) : nullptr);
            clear_nans <<<blockNum(n*4), threadsPerBlock>>> ((real*)d_dpp, (real*)d_dpp, n*4);
            break;
        default:
            break;
        }
        
        sta = rdot(cub_hdl, n2 * 2, (real*)d_dpp, 1, (real*)d_grad, 1, dgdotdpp_and_norm2dpp);
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        sta = rdot(cub_hdl, n2 * 2, (real*)d_dpp, 1, (real*)d_dpp, 1, dgdotdpp_and_norm2dpp+1);  // compute norm square of the step 
        ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

        print_gpu_value<real,2>(dgdotdpp_and_norm2dpp, "dgdotdpp_norm2dpp");

        if(optimization_method<OM_NEWTON || optimization_method>OM_NEWTON_SPDH_FULLEIG){
            conjugate_inplace <<<blockNum(n), threadsPerBlock >>> (d_dpp + n, n);
            CUDA_CHECK_ERROR;
        }

        // dfzgz = D2*dpp
        HM.update_fzgz(d_dfzgz, d_dpp);

        // dfgP2P = C2*dpp
        HM.update_fgP2P(d_dfgP2P, d_dpp);

        HM.compute_max_step_to_preserve_orientation(step, d_dfzgz);
        print_gpu_value(step, "maxtForPhiPsy");

        harmonic_line_search << <1, enEvalsPerKernel >> > (d_fzgz0, d_dfzgz, d_fgP2P, d_dfgP2P, validationData.nextSampleInSameCage, d_bP2P, m, nP2P, step,
            d_en, isoEnergyType, isoepow, dgdotdpp_and_norm2dpp, lambda, enEvalsPerKernel);
        print_gpu_value(step, "step size after line search");

        if (needValidateMap) {
            sta = gemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, m, 2, n, pOne, validationData.E2, m, d_dpp, n, pZero, fzzgzz.data() + m * 2, m);
            ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

            abs_diff_similarity_polygon << <blockNum(nAllVars), threadsPerBlock >> > (abs_dS0dSt0.data(), d_phipsy, d_dpp, validationData.v, nAllVars, step);

            sta = rgemm(cub_hdl, CUBLAS_OP_N, CUBLAS_OP_N, m, 4, n, (const real*)pOne, validationData.L, m, abs_dS0dSt0.data(), nAllVars, (const real*)pZero, delta_L_fzgz.data(), m);
            ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");

            real *norm2dpp = dgdotdpp_and_norm2dpp + 1;
            harmonic_map_validate_bounds << <1, 1 >> > (d_fzgz0, d_dfzgz, m, fzzgzz.data(), fzzgzz.data() + m * 2, delta_L_fzgz.data(), sampleSpacings,
                validationData.nextSampleInSameCage, step, distortion_bounds.data() + it * 6, norm2dpp, enEvalsPerKernel);

            if (it < nIter - 1) {  // update dfzzgzz
                sta = axpy(cub_hdl, m * 4, step, (const real*)(fzzgzz.data() + m * 2), 1, (real*)fzzgzz.data(), 1);
                ensure(CUBLAS_STATUS_SUCCESS == sta, "CUBLAS error");
            }

            myZeroFill(d_en);
            fMyEnergy(d_dfzgz, step, d_dfgP2P, d_en);  // update energy
        }
        print_gpu_value(step, "step size after validation");

        if (reportIterationStats) {
            real* iter_stats = stats_energies_steps.data() + (1+it) * 3;
            stepNorm_from_stepSize_sqrNorm <<<1, 1>>> (step, dgdotdpp_and_norm2dpp + 1, iter_stats);
            p2p_energy<Complex, real> <<<blockNum(nP2P), threadsPerBlock >>> (iter_stats + 1, d_fgP2P, d_dfgP2P, step, d_bP2P, nP2P, lambda);
            myCopy_n(d_en, 1, iter_stats + 2);
        }


        recordEvents(curIterTimes + TIME_LS, 1);

        //////////////////////////////////////////////////////////////////////////
        HM.increment_phipsy(d_dpp, step);
        if (it < nIter - 1) {
            HM.increment_fzgz(d_dfzgz, step);
            HM.increment_fgP2P(d_dfgP2P, step);
        }

        if (it==nIter-1) {
            myCopy_n(d_phipsyIters, n2, d_phipsyIters + n2);
            myCopy_n(d_phipsy, n2, d_phipsyIters);
        }

        recordEvents(curIterTimes + TIME_END, 1);
    }

    real e = copyValFromGPU(d_en);

    if (needValidateMap) {
        const std::vector<real> bounds_all = distortion_bounds;

        for (int i = 0; i < nIter; i++) {
            const auto bounds = bounds_all.data() + i * 6;
            printf("It%2d, sig1: (%.3f,%.3f)   sig2: (%.3f,%.3f)   k: (%.3f,%.3f)\n", i, bounds[0], bounds[3], bounds[1], bounds[4], bounds[2], bounds[5]);
        }
    }

    std::vector<double> allStats(1, e);
    if (reportIterationStats) {
        //cudaEventSynchronize(times.back()); // not necessary after having d_en copied to cpu, however be careful not to have it optimized when the value is not used
        const std::vector<real> iterStats = stats_energies_steps;

        allStats.resize((nIter + 1) * 8);

        const char* headers[] = { "Iter", "Grad", "Hess", "Solve", "LS", "All", "|step|", "E_p2p", "E_all" };
        printf("%5s %6s %6s %6s %6s %6s %9s %9s %9s\n", headers[0], headers[1], headers[2], headers[3], headers[4], headers[5], headers[6], headers[7], headers[8]);
        float ittimes[TIME_NUMS - 1] = { 0 };

        for (int i = 0; i < nIter+1; i++) {
            printf("%5d %6.2f %6.2f %6.2f %6.2f %6.2f %9.2e %9.2e %9.2e\n", i, ittimes[0], ittimes[1], ittimes[2], ittimes[3], ittimes[4], iterStats[i*3], iterStats[i*3+1], iterStats[i*3+2]);

            std::copy_n(ittimes, TIME_NUMS - 1, &allStats[i * 8]);
            std::copy_n(&iterStats[i * 3], 3, &allStats[i * 8 + 5]);

            if (i == nIter) continue;

            for (int j = 0; j < TIME_END - 1; j++)
                cudaEventElapsedTime(ittimes + j, times[i*TIME_NUMS + j], times[i*TIME_NUMS + j + 1]);

            cudaEventElapsedTime(ittimes + TIME_END-1, times[i*TIME_NUMS + TIME_BEGIN], times[i*TIME_NUMS + TIME_END]);
        }

        for (auto &tt : times) cudaEventDestroy(tt);
    }

    return allStats;
}
