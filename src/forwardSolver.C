
#include "mpi.h"
#include "omg.h"
#include "oda.h"
#include "TreeNode.h"
#include "parUtils.h"
#include "sys.h"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

#include "dendro.h"
#include "externVars.h"

namespace ot {
  extern double**** ShapeFnCoeffs;
}

void myNewton(ot::DAMG* damg, double fTol, double xTol, 
    int maxIterCnt, Vec sol);

void createGaussPtsAndWts(double*& gPts, double*& gWts, int numGpts);
void destroyGaussPtsAndWts(double*& gPts, double*& gWts);

void ComputeResidual(ot::DAMG damg, double kappa, int numGpts, double* gPts, double* gWts, Vec in, Vec out); 

void computeSinhTerm(ot::DA* da, double kappa, int numGpts, double* gPts, double* gWts, Vec in, Vec out);

void computeLaplacianTerm(ot::DA* da, int numGpts, double* gPts, double* gWts, Vec in, Vec out);

void computeRobinTerm(ot::DA* da, int numGpts, double* gPts, double* gWts, Vec in, Vec out);

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

  double kappa = 1.0;

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

void ComputeResidual(ot::DAMG damg, double kappa, int numGpts, double* gPts, double* gWts, Vec in, Vec out) {
  Vec sinhTerm;
  Vec robTerm;

  VecDuplicate(out, &sinhTerm);
  VecDuplicate(out, &robTerm);

  ot::DA* da = damg->da;

  computeLaplacianTerm(da, numGpts, gPts, gWts, in, out);
  computeSinhTerm(da, kappa, numGpts, gPts, gWts, in, sinhTerm);
  computeRobinTerm(da, numGpts, gPts, gWts, in, robTerm);

  VecAXPBYPCZ(out, 1.0, -1.0, 1.0, sinhTerm, robTerm);

  VecDestroy(sinhTerm);
  VecDestroy(robTerm);
}

void computeSinhTerm(ot::DA* da, double kappa, int numGpts, double* gPts, double* gWts, Vec in, Vec out) {
  PetscScalar *inarray;
  PetscScalar *outarray;
  VecZeroEntries(out);
  //Nodal, Non-Ghosted, Read-only, 1-dof
  da->vecGetBuffer(in, inarray, false, false, true, 1);

  //Nodal, Non-Ghosted, Writable, 1-dof
  da->vecGetBuffer(out, outarray, false, false, false, 1);

  unsigned int maxD;
  unsigned int balOctmaxD;

  if(da->iAmActive()) {
    da->ReadFromGhostsBegin<PetscScalar>(inarray, 1);
    da->ReadFromGhostsEnd<PetscScalar>(inarray);

    maxD = da->getMaxDepth();
    balOctmaxD = maxD - 1;
    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>())  
    {
      unsigned int idx = da->curr();
      unsigned levelhere = (da->getLevel(idx) - 1);
      double hxOct = (double)((double)(1u << (balOctmaxD - levelhere))/(double)(1u << balOctmaxD));
      double fac = ((hxOct*hxOct*hxOct)/8.0);
      unsigned int indices[8];
      da->getNodeIndices(indices); 
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        //Quadrature Rule
        for(int m = 0; m < numGpts; m++) {
          for(int n = 0; n < numGpts; n++) {
            for(int p = 0; p < numGpts; p++) {
              double inVal = 0.0;                
              for(unsigned int k = 0; k < 8; k++) {
                double ShFnVal_k = ( ot::ShapeFnCoeffs[childNum][elemType][k][0] + 
                    (ot::ShapeFnCoeffs[childNum][elemType][k][1]*gPts[m]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][k][2]*gPts[n]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][k][3]*gPts[p]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][k][4]*gPts[m]*
                     gPts[n]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][k][5]*gPts[n]*
                     gPts[p]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][k][6]*gPts[p]*
                     gPts[m]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][k][7]*gPts[m]*
                     gPts[n]*gPts[p]) );

                inVal += (inarray[indices[k]]*ShFnVal_k); 
              }//end for k
              double rhsVal = fac*kappa*kappa*sinh(inVal);
              for(unsigned int j = 0; j < 8; j++) {
                double ShFnVal_j = ( ot::ShapeFnCoeffs[childNum][elemType][j][0] + 
                    (ot::ShapeFnCoeffs[childNum][elemType][j][1]*gPts[m]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][2]*gPts[n]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][3]*gPts[p]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][4]*gPts[m]*
                     gPts[n]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][5]*gPts[n]*
                     gPts[p]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][6]*gPts[p]*
                     gPts[m]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][7]*gPts[m]*
                     gPts[n]*gPts[p]) );

                outarray[indices[j]] += (gWts[m]*gWts[n]
                    *gWts[p]*rhsVal*ShFnVal_j);
              }//end for j
            }//end for p
          }//end for n
        }//end for m
    }//end ALL loop 
  }//end if active

  //Nodal, Non-Ghosted, Read-only, 1-dof
  da->vecRestoreBuffer(in, inarray, false, false, true, 1);

  //Nodal, Non-Ghosted, Writable, 1-dof
  da->vecRestoreBuffer(out, outarray, false, false, false, 1);
}


void computeLaplacianTerm(ot::DA* da, int numGpts, double* gPts, double* gWts, Vec in, Vec out) {
  PetscScalar *inarray;
  PetscScalar *outarray;
  VecZeroEntries(out);
  //Nodal, Non-Ghosted, Read-only, 1-dof
  da->vecGetBuffer(in, inarray, false, false, true, 1);

  //Nodal, Non-Ghosted, Writable, 1-dof
  da->vecGetBuffer(out, outarray, false, false, false, 1);

  unsigned int maxD;
  unsigned int balOctmaxD;

  if(da->iAmActive()) {
    da->ReadFromGhostsBegin<PetscScalar>(inarray, 1);
    da->ReadFromGhostsEnd<PetscScalar>(inarray);

    maxD = da->getMaxDepth();
    balOctmaxD = maxD - 1;
    for(da->init<ot::DA_FLAGS::ALL>();
        da->curr() < da->end<ot::DA_FLAGS::ALL>();
        da->next<ot::DA_FLAGS::ALL>())  
    {
      unsigned int idx = da->curr();
      unsigned levelhere = (da->getLevel(idx) - 1);
      double hxOct = (double)((double)(1u << (balOctmaxD - levelhere))/(double)(1u << balOctmaxD));
      double fac = (hxOct/2.0);
      unsigned int indices[8];
      da->getNodeIndices(indices); 
      unsigned char childNum = da->getChildNumber();
      unsigned char hnMask = da->getHangingNodeIndex(idx);
      unsigned char elemType = 0;
      GET_ETYPE_BLOCK(elemType,hnMask,childNum)
        //Quadrature Rule
        for(int m = 0; m < numGpts; m++) {
          for(int n = 0; n < numGpts; n++) {
            for(int p = 0; p < numGpts; p++) {
              double gradPhi[8][3];
              for(int j = 0; j < 8; j++) {
                gradPhi[j][0] = ( (ot::ShapeFnCoeffs[childNum][elemType][j][1]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][4]*gPts[n]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][6]*gPts[p]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][7]*gPts[n]*gPts[p]) );

                gradPhi[j][1] = ( (ot::ShapeFnCoeffs[childNum][elemType][j][2]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][4]*gPts[m]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][5]*gPts[p]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][7]*gPts[m]*gPts[p]) );

                gradPhi[j][2] = ( (ot::ShapeFnCoeffs[childNum][elemType][j][3]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][5]*gPts[n]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][6]*gPts[m]) +
                    (ot::ShapeFnCoeffs[childNum][elemType][j][7]*gPts[m]*gPts[n]) );
              }//end for j
              for(int j = 0; j < 8; j++) {
                for(int i = 0; i < 8; i++) {
                  outarray[indices[j]] += (gWts[m]*gWts[n]*gWts[p]*fac*inarray[indices[i]]*(
                        (gradPhi[i][0]*gradPhi[j][0]) + 
                        (gradPhi[i][1]*gradPhi[j][1]) + 
                        (gradPhi[i][2]*gradPhi[j][2]) ));
                }//end for i
              }//end for j
            }//end for p
          }//end for n
        }//end for m
    }//end ALL loop 
  }//end if active

  //Nodal, Non-Ghosted, Read-only, 1-dof
  da->vecRestoreBuffer(in, inarray, false, false, true, 1);

  //Nodal, Non-Ghosted, Writable, 1-dof
  da->vecRestoreBuffer(out, outarray, false, false, false, 1);
}

void computeRobinTerm(ot::DA* da, int numGpts, double* gPts, double* gWts, Vec in, Vec out) {
  VecZeroEntries(out);
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



