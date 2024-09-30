
#include <vector>

void loadBCRSMatrixVectors(std::vector<double>& vecVals, std::vector<int>& vecCols, std::vector<int>& vecRows, const std::string& filename);

void loadBCRSMatrixVectors(std::vector<double>& vecVals, std::vector<unsigned int>& vecCols, std::vector<unsigned int>& vecRows, const std::string& filename);

std::vector<double> loadResVector(const std::string& filename);


/*
  std::cout << "Dvals: ";
  std::cout << size(Dvals) << " " <<std::endl;
  //for (const auto& val : Dvals) std::cout << val << " ";
  //std::cout << std::endl;
  std::cout << "Dcols: ";
  std::cout << size(Dcols) << " " <<std::endl;
  for (const auto& val : Dcols) std::cout << val << " ";
  std::cout << std::endl;
  std::cout << "Drows: ";
  std::cout << size(Drows) << " " << std::endl;
  //for (const auto& val : Drows) std::cout << val << " ";
  //std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Bvals: ";
  std::cout << size(Bvals) << " " <<std::endl;
  for (const auto& val : Bvals) std::cout << val << " ";
  std::cout << std::endl;
  std::cout << "Bcols: ";
  std::cout << size(Bcols) << " " <<std::endl;
  for (const auto& val : Bcols) std::cout << val << " ";
  std::cout << std::endl;
  std::cout << "Brows: ";
  std::cout << size(Brows) << " " <<std::endl;
  for (const auto& val : Brows) std::cout << val << " ";
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Cvals: ";
  std::cout << size(Cvals) << " " <<std::endl;
  std::cout << "Ccols: ";
  std::cout << size(Ccols) << " " <<std::endl;
  for (const auto& val : Ccols) std::cout << val << " ";
  std::cout << std::endl;
  std::cout << "Crows: ";
  std::cout << size(Crows) << " " <<std::endl;
  for (const auto& val : Crows) std::cout << val << " ";
  std::cout << std::endl;
  std::cout << std::endl;
*/
