
#include <vector>

void loadBCRSMatrixVectors(std::vector<double>& vecVals, std::vector<int>& vecCols, std::vector<int>& vecRows, const std::string& filename);

void loadBCRSMatrixVectors(std::vector<double>& vecVals, std::vector<unsigned int>& vecCols, std::vector<unsigned int>& vecRows, const std::string& filename);

std::vector<double> loadResVector(const std::string& filename);
