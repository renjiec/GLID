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
    std::vector<std::string> solver_names;    //const char *[] = { "newton", ... "CVX", "Direct Mosek" };
    std::vector<std::string> energy_names;    //const char *[] = { 'ARAP', 'BARAP', 'ISO', 'EISO', 'AMIPS', 'BETA'};
    int solver;
    int energy_type;
    float p2p_weight;

    bool softP2P;
    int AQP_IterPerUIUpdate;
    Eigen::MatrixXcd C; // Cauchy Coordiantes for vertices

    MyMesh &M;

    MatlabDeformer(MatlabDeformer&) = delete;

    MatlabDeformer(MyMesh &m) :M(m), solver(0), 
        p2p_weight(100000.f), 
        softP2P(true), AQP_IterPerUIUpdate(10) {

        using deformerptr = MatlabDeformer*;

        TwBar *bar = TwNewBar("ShapeDeformer");

        TwDefine(" ShapeDeformer size='220 180' color='255 0 255' text=dark alpha=128 position='5 380' label='Shape Deformer'"); // change default tweak bar size and color

        //////////////////////////////////////////////////////////////////////////

        solver_names = matlab2strings("harmonic_map_solvers");
        std::string defaultsolver = matlab2string("default_harmonic_map_solver");
        solver = 0;
        for (int i = 0; i < solver_names.size(); i++) {
            if (defaultsolver == solver_names[i]) solver = i;
        }

        energy_names = matlab2strings("harmonic_map_energies");
        std::string defaultenergy = matlab2string("default_harmonic_map_energy");
        energy_type = 0;
        for (int i = 0; i < energy_names.size(); i++) {
            if (defaultenergy == energy_names[i]) energy_type = i;
        }

        TwAddButton(bar, "Clear P2P", [](void *d) { 
            deformerptr(d)->M.constrainVertices.clear(); 
            deformerptr(d)->M.actConstrainVertex = -1; 
            deformerptr(d)->updateP2PConstraints(-1); },
            this, " ");

        TwAddButton(bar, "Reset Mesh", [](void *d) {
            deformerptr(d)->resetDeform(); 
            deformerptr(d)->M.mMeshScale = 1.f;
            deformerptr(d)->M.mTranslate.assign(0.f);
            matlabEval("AQP_preprocessed = false; P2P_Deformation_Converged = 0;"); }, this, " key=r ");

        //////////////////////////////////////////////////////////////////////////

        TwAddVarRW(bar, "P2P weight", TW_TYPE_FLOAT, &p2p_weight, " ");

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

        preprocess();
    }

    ~MatlabDeformer(){
        TwBar *bar = TwGetBarByName("ShapeDeformer");
        if (bar)    TwDeleteBar(bar); 
    }

    virtual std::string name(){ return "P2PHarmonic"; }

    virtual void preprocess() 
    {
        matlabEval("p2p_harmonic_prep;");

        C = matlab2eigenComplex("C");
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
		if ( !resVarName.compare("PhiPsy") ) { // do all interpolation computation in matlab, for better performance with # virtual vertex > 1
			MatrixXcd Phi = matlab2eigenComplex("Phi");
			MatrixXcd Psy = matlab2eigenComplex("Psy");

			if (Phi.rows() == 0 || Psy.rows() == 0 || C.rows() == 0) return;

			Eigen::VectorXcd x = C*Phi + (C*Psy).conjugate();

			if (getMatEngine().hasVar("rot_trans")) {
				// for interpolation
				Vector2cd rot_trans = matlab2eigenComplex("rot_trans");
				x = x.array()*rot_trans(0) + rot_trans(1);
			}

			if (x.rows() == 0) return;

			Matrix<float, Dynamic, 2, RowMajor> xr(x.rows(), 2);
			xr.col(0) = x.real().cast<float>();
			xr.col(1) = x.imag().cast<float>();
			M.upload(xr, Eigen::MatrixXi(), nullptr);
		}
		else {
			MatrixXcd x = matlab2eigenComplex(resVarName);
			if (x.rows() == 0) return;

			Matrix<float, Dynamic, 2, RowMajor> xr(x.rows(), 2);
			xr.col(0) = x.real().cast<float>();
			xr.col(1) = x.imag().cast<float>();
			M.upload(xr, Eigen::MatrixXi(), nullptr);
		}
    }

    virtual void deform()
    {
        string2matlab("solver_type", solver_names[solver]);
        string2matlab("energy_type", energy_names[energy_type]);
        scalar2matlab("p2p_weight", p2p_weight);
        matlabEval("p2p_harmonic;");
        matlabEval("clear rot_trans;");

        deformResultFromMaltab("XP2PDeform");
    }

    virtual bool converged() {
        return matlab2scalar("P2P_Deformation_Converged", 0) > 0;
    }

    virtual void resetDeform() {
		matlabEval("Phi = vv; Psy = Phi * 0; XP2PDeform = X; phipsyIters = []; clear rot_trans;");
        deformResultFromMaltab("XP2PDeform"); 
    }
    virtual void getResult() {}
    virtual void saveData()   { matlabEval("p2p_harmonic_savedata;"); }
};
