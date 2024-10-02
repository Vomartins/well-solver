
#ifndef READ_VECTORS_HPP
#define READ_VECTORS_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <limits.h>

template <typename I>
void loadSparseMatrixVectors(std::vector<double>& vecVals, std::vector<I>& vecCols, std::vector<I>& vecRows, const std::string& filename);

std::vector<double> loadResVector(const std::string& filename);

void squareCSCtoCSR(std::vector<double> Dvals, std::vector<int> Drows, std::vector<int> Dcols, std::vector<double>& Dvals_, std::vector<int>& Drows_, std::vector<int>& Dcols_);

#endif
