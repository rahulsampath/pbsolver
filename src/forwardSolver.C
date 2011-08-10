
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

int main(int argc, char** argv) {
  bool incCorner = 1;  
  unsigned int maxNumPts = 1;
  unsigned int dim = 3;
  unsigned int maxDepth = 30;
  bool compressLut = false;
  double mgLoadFac = 2.0;
  unsigned int   dof = 1; // degrees of freedom per node  
  int       nlevels = 10; //number of multigrid levels

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

  DAMGDestroy(damg);

  ot::DAMG_Finalize();
  PetscFinalize();
}

