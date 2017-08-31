#pragma once

#include "vaomesh.h"

struct Deformer
{
    bool needIteration = true;
    virtual std::string name(){ return "UNKNOWN"; }
    virtual void preprocess() {}
    virtual void updateP2PConstraints(int){}
    virtual void deform() = 0;
    virtual bool converged() { return false; }
    virtual void resetDeform(){}
    virtual void getResult(){}
    virtual void saveData(){}
};
