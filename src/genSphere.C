
#include "octUtils.h"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

#include "dendro.h"
#include "externVars.h"

#define __PI__ 3.1415926535897932

void rescalePts(std::vector<double> & pts);

int main(int argc, char** argv) {

  if(argc < 4) {
    std::cout<<"exe radius numPts ptsFile"<<std::endl;
    exit(0);
  }

  double radius = atof(argv[1]);
  int numPts = atoi(argv[2]);

  std::vector<double> pts;

  int numIntervals = static_cast<int>(floor(sqrt(0.5*static_cast<double>(numPts))));
  std::cout<<"numIntervals = "<<numIntervals<<std::endl;  

  for(int i = 0; i < numIntervals; i++) {
    double u = __PI__*(static_cast<double>(i))/
      (static_cast<double>(numIntervals));

    for(int j = 0; j < numIntervals; j++) {
      double v = __PI__*(static_cast<double>(j))/
        (static_cast<double>(numIntervals));

      pts.push_back(radius*cos(u)*cos(v));
      pts.push_back(radius*cos(u)*sin(v));
      pts.push_back(radius*sin(u));
    }
  }

  std::cout<<"True total number of points = "<<(pts.size()/3)<<std::endl;

  rescalePts(pts);

  ot::writePtsToFile(argv[3], pts);

}

void rescalePts(std::vector<double> & pts) 
{
  double minX = pts[0];
  double maxX = pts[0];

  double minY = pts[1];
  double maxY = pts[1];

  double minZ = pts[2];
  double maxZ = pts[2];

  for(unsigned int i = 0; i < pts.size(); i += 3) {
    double xPt = pts[i];
    double yPt = pts[i + 1];
    double zPt = pts[i + 2];

    if(xPt < minX) {
      minX = xPt;
    }

    if(xPt > maxX) {
      maxX = xPt;
    }

    if(yPt < minY) {
      minY = yPt;
    }

    if(yPt > maxY) {
      maxY = yPt;
    }

    if(zPt < minZ) {
      minZ = zPt;
    }

    if(zPt > maxZ) {
      maxZ = zPt;
    }
  }//end for i

  double xRange = (maxX - minX);
  double yRange = (maxY - minY);
  double zRange = (maxZ - minZ);

  for(unsigned int i = 0; i < pts.size();  i += 3) {
    double xPt = pts[i];
    double yPt = pts[i + 1];
    double zPt = pts[i + 2];

    pts[i] = 0.025 + (0.95*(xPt - minX)/xRange);
    pts[i + 1] = 0.025 + (0.95*(yPt - minY)/yRange);
    pts[i + 2] = 0.025 + (0.95*(zPt - minZ)/zRange);
  }//end for i

}


