#pragma once

#include "deformer.h"
#include "matlab_utils.h"
#include <thread>
#include <chrono>

extern bool showATB;
extern int viewport[4];
void display();
void loadP2PConstraints();

inline std::string catStr(const std::vector<std::string> &names)
{
    std::string str;
    for (int i = 0; i < names.size(); i++) {
        str += names[i];
        if (i < names.size() - 1) str += ", ";
    }
    return str;
}

struct MatlabDeformer : public Deformer
{
    std::vector<std::string> solver_names;    //const char *[] = { "newton", ... };
    std::vector<std::string> energy_names;    //const char *[] = { 'ISO', 'EISO', 'AMIPS' };
    int solver = 0;
    int energy_type = 0;

    MyMesh &M;

    MatlabDeformer(MatlabDeformer&) = delete;

    MatlabDeformer(MyMesh &m) :M(m){

        using deformerptr = MatlabDeformer*;

        TwBar *bar = TwNewBar("GLIDDeformer");

        TwDefine(" GLIDDeformer size='220 180' color='255 0 255' text=dark alpha=128 position='5 300' label='GLID Deformer'"); // change default tweak bar size and color

        //////////////////////////////////////////////////////////////////////////
        solver_names = matlab2strings("harmonic_map_solvers");
        std::string defaultsolver = matlab2string("default_harmonic_map_solver");
        for (int i = 0; i < solver_names.size(); i++) if (defaultsolver == solver_names[i]) solver = i;

        energy_names = matlab2strings("harmonic_map_energies");
        std::string defaultenergy = matlab2string("default_harmonic_map_energy");
        for (int i = 0; i < energy_names.size(); i++) if (defaultenergy == energy_names[i]) energy_type = i;

        TwType energyType = TwDefineEnumFromString("Energy", catStr(energy_names).c_str());
        TwAddVarRW(bar, "Energy", energyType, &energy_type, " ");
 
        TwType solverType = TwDefineEnumFromString("Solver", catStr(solver_names).c_str());
        TwAddVarRW(bar, "Solver", solverType, &solver, " ");


        //////////////////////////////////////////////////////////////////////////
        TwAddVarCB(bar, "P2P weight", TW_TYPE_FLOAT, 
            [](const void *v, void *) { scalar2matlab("p2p_weight", *(const float*)(v)); },
            [](void *v, void *) { *(float*)(v) = matlab2scalar("p2p_weight"); },
            nullptr, " min=0 ");

        TwAddVarCB(bar, "#Samples", TW_TYPE_INT32, 
            [](const void *v, void *d) { scalar2matlab("numEnergySamples", *(const int*)(v)); 
                matlabEval("p2p_harmonic_prep;");
                },
            [](void *v, void *) { *(int*)(v) = matlab2scalar("numEnergySamples"); },
            nullptr, " min=100 ");

        TwAddVarCB(bar, "energy param", TW_TYPE_FLOAT, 
            [](const void *v, void *d) { scalar2matlab("energy_parameter", *(const float*)(v)); },
            [](void *v, void *)       { *(float*)(v) = matlab2scalar("energy_parameter"); },
            nullptr, " min=0 help='change the parameter for some energies, e.g. AMIPS, BARAP, power iso' ");

        TwAddButton(bar, "Reset View", [](void *d) {
            deformerptr(d)->M.updateBBox();
            deformerptr(d)->M.mMeshScale = 1.f;
            deformerptr(d)->M.mTranslate.assign(0.f);
        }, this, " ");

        TwAddButton(bar, "Reset Shape", [](void *d) {
            deformerptr(d)->resetDeform(); 
            matlabEval("NLO_preprocessed = false; P2P_Deformation_Converged = 0;"); }, this, " key=r ");

        TwAddVarCB(bar, "Pause", TW_TYPE_BOOLCPP,
            [](const void *v, void *d) {  deformerptr(d)->needIteration = !*(bool*)(v); },
            [](void *v, void *d) { *(bool*)(v) = !deformerptr(d)->needIteration; },
            this, " key=i ");

        TwAddButton(bar, "Clear P2P", [](void *d) { 
            deformerptr(d)->M.constrainVertices.clear(); 
            deformerptr(d)->M.actConstrainVertex = -1; 
            deformerptr(d)->updateP2PConstraints(-1); },
            this, " ");

        preprocess();
    }

    ~MatlabDeformer(){
        TwBar *bar = TwGetBarByName("GLIDDeformer");
        if (bar)    TwDeleteBar(bar); 
    }

    virtual std::string name(){ return "P2PHarmonic"; }

    virtual void preprocess() 
    {
        matlabEval("p2p_harmonic_prep;");

        deformResultFromMaltab("XP2PDeform");
    }


    virtual void updateP2PConstraints(int) 
    {
        using namespace Eigen;
        const size_t nConstrain = M.constrainVertices.size();
        eigen2matlab("P2PVtxIds", (Map<VectorXi>(M.getConstrainVertexIds().data(), nConstrain) + VectorXi::Ones(nConstrain)).cast<double>());
        matlabEval("CauchyCoordinatesAtP2Phandles = C(P2PVtxIds,:);");

        MatrixX2d p2pdst = Map<Matrix<float, Dynamic, 2, RowMajor> >(M.getConstrainVertexCoords().data(), nConstrain, 2).cast<double>();
        eigen2matlabComplex("P2PCurrentPositions", p2pdst.col(0), p2pdst.col(1));
    }


    void deformResultFromMaltab(std::string resVarName)
    {
        using namespace Eigen;
        MatrixXcd x = matlab2eigenComplex(resVarName);
        if (x.rows() == 0) return;

        Matrix<float, Dynamic, 2, RowMajor> xr(x.rows(), 2);
        xr.col(0) = x.real().cast<float>();
        xr.col(1) = x.imag().cast<float>();
        M.upload(xr, Eigen::MatrixXi(), nullptr);
    }

    virtual void deform()
    {
        string2matlab("solver_type", solver_names[solver]);
        string2matlab("energy_type", energy_names[energy_type]);
        matlabEval("p2p_harmonic;");

        deformResultFromMaltab("XP2PDeform");
    }

    virtual bool converged() {
        return !getMatEngine().hasVar("P2P_Deformation_Converged")  ||  matlab2scalar("P2P_Deformation_Converged") > 0;
    }

    virtual void resetDeform() {
		matlabEval("Phi = vv; Psy = Phi * 0; XP2PDeform = X; phipsyIters = [];");
        deformResultFromMaltab("XP2PDeform"); 
    }
    virtual void getResult() {}
    virtual void saveData()   { matlabEval("p2p_harmonic_savedata;"); }
};
