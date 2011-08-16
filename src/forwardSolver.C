
#include "mpi.h"
#include "omg.h"
#include "TreeNode.h"
#include "parUtils.h"
#include "sys.h"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

#include "dendro.h"
#include "externVars.h"

void myNewton(ot::DAMG* damg, double fTol, double xTol, 
    int maxIterCnt, Vec sol);

void createGaussPtsAndWts(double*& gPts, double*& gWts, int numGpts);
void destroyGaussPtsAndWts(double*& gPts, double*& gWts);

int main(int argc, char** argv) {
  bool incCorner = 1;  
  unsigned int maxNumPts = 1;
  unsigned int dim = 3;
  unsigned int maxDepth = 30;
  bool compressLut = false;
  double mgLoadFac = 2.0;
  unsigned int   dof = 1; // degrees of freedom per node  
  int       nlevels = 10; //number of multigrid levels

  if(argc < 2) {
    std::cout<<"exe ptsFile"<<std::endl;
    exit(0);
  }

  PetscInitialize(&argc, &argv, 0, 0);
  ot::RegisterEvents();
  ot::DAMG_Initialize(MPI_COMM_WORLD);

  std::vector<double> pts;
  ot::readPtsFromFile(argv[1], pts);

  double gSize[3];
  gSize[0] = 1.0;
  gSize[1] = 1.0;
  gSize[2] = 1.0;

  std::vector<ot::TreeNode> linOct, balOct;
  ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, MPI_COMM_WORLD);

  std::cout<<"linOct size = "<<linOct.size()<<std::endl;

  ot::balanceOctree (linOct, balOct, dim, maxDepth, incCorner, MPI_COMM_WORLD, NULL, NULL);

  std::cout<<"balOct size = "<<balOct.size()<<std::endl;

  ot::DAMG       *damg;    
  ot::DAMGCreateAndSetDA(MPI_COMM_WORLD, nlevels, NULL, &damg, 
      balOct, dof, mgLoadFac, compressLut, incCorner);

  ot::PrintDAMG(damg);

  PetscInt numGpts = 4;
  double* gPts;
  double* gWts;
  createGaussPtsAndWts(gPts, gWts, numGpts);
  
  double fTol = 1.0e-10;
  double xTol = 1.0e-10;
  int maxIterCnt = 10;
  Vec sol;
  VecDuplicate(DAMGGetx(damg), &sol);
  // myNewton(damg, fTol, xTol, maxIterCnt, sol);

  DAMGDestroy(damg);
  VecDestroy(sol);
  destroyGaussPtsAndWts(gPts, gWts);

  ot::DAMG_Finalize();

  std::cout<<"Done."<<std::endl;

  PetscFinalize();
}

void myNewton(ot::DAMG* damg, double fTol, double xTol, 
    int maxIterCnt, Vec sol) {

  double stepFactor = 1.0;

  //1. Evaluate residual function using (sol and DAMGGetRHS) 

  double resNorm = 0;
  int iterCnt = 0;
  while ( (resNorm > fTol) && (iterCnt < maxIterCnt) ) {

    //2. evaluate jacobian via DAMGSetKSP

    //3. DAMGSolve

    //4. Update solution

    //5. evaluate residual function
    iterCnt++;
  }

}

void createGaussPtsAndWts(double*& gPts, double*& gWts, int numGpts) {
  gPts = new double[numGpts];
  gWts = new double[numGpts];

  if(numGpts == 3) {
    //3-pt rule
    gWts[0] = 0.88888889;  gWts[1] = 0.555555556;  gWts[2] = 0.555555556;
    gPts[0] = 0.0;  gPts[1] = 0.77459667;  gPts[2] = -0.77459667;
  } else if(numGpts == 4) {
    //4-pt rule
    gWts[0] = 0.65214515;  gWts[1] = 0.65214515;
    gWts[2] = 0.34785485; gWts[3] = 0.34785485;  
    gPts[0] = 0.33998104;  gPts[1] = -0.33998104;
    gPts[2] = 0.86113631; gPts[3] = -0.86113631;
  } else if(numGpts == 5) {
    //5-pt rule
    gWts[0] = 0.568888889;  gWts[1] = 0.47862867;  gWts[2] =  0.47862867;
    gWts[3] = 0.23692689; gWts[4] = 0.23692689;
    gPts[0] = 0.0;  gPts[1] = 0.53846931; gPts[2] = -0.53846931;
    gPts[3] = 0.90617985; gPts[4] = -0.90617985;
  } else if(numGpts == 6) {
    //6-pt rule
    gWts[0] = 0.46791393;  gWts[1] = 0.46791393;  gWts[2] = 0.36076157;
    gWts[3] = 0.36076157; gWts[4] = 0.17132449; gWts[5] = 0.17132449;
    gPts[0] = 0.23861918; gPts[1] = -0.23861918; gPts[2] = 0.66120939;
    gPts[3] = -0.66120939; gPts[4] = 0.93246951; gPts[5] = -0.93246951;
  } else if(numGpts == 7) {
    //7-pt rule
    gWts[0] = 0.41795918;  gWts[1] = 0.38183005; gWts[2] = 0.38183005;
    gWts[3] = 0.27970539;  gWts[4] = 0.27970539; 
    gWts[5] = 0.12948497; gWts[6] = 0.12948497;
    gPts[0] = 0.0;  gPts[1] = 0.40584515;  gPts[2] = -0.40584515;
    gPts[3] = 0.74153119;  gPts[4] = -0.74153119;
    gPts[5] = 0.94910791; gPts[6] = -0.94910791;
  } else  {
    assert(false);
  }
}

void destroyGaussPtsAndWts(double*& gPts, double*& gWts) {
  assert(gPts);
  delete [] gPts;
  gPts = NULL;

  assert(gWts);
  delete [] gWts;
  gWts = NULL;
}



