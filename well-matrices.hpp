
#include <dune/common/version.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <fstream>


class WellMatrices
{
public:
  const static int dim = 3;
  const static int dim_wells = 4;

  // Matrix type for matrix D
  using DiagMatrixBlockWellType = Dune::FieldMatrix<double,dim_wells,dim_wells>;
  using DiagMatWell = Dune::BCRSMatrix<DiagMatrixBlockWellType>;

  // Matrix type for matrices D and C^t
  using OffDiagMatrixBlockWellType = Dune::FieldMatrix<double,dim_wells,dim>;
  using OffDiagMatWell = Dune::BCRSMatrix<OffDiagMatrixBlockWellType>;

  DiagMatWell duneD;
  OffDiagMatWell duneB, duneC;

  const char* fileName_D = "matrix-D.mm";
  const char* fileName_B = "matrix-B.mm";
  const char* fileName_C = "matrix-C.mm";

  //std::ifstream fileIn_D(fileName_D);
  //std::ifstream fileIn_B(fileName_B);
  //std::ifstream fileIn_C(fileName_C);

public:
    void read_matrices();
};
