
#include "read-vectors.hpp"

#include <iostream>
#include <fstream>

void loadBCRSMatrixVectors(std::vector<double>& vecVals, std::vector<int>& vecCols, std::vector<int>& vecRows, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading." << std::endl;
        return;
    }

    // Read values vector
    size_t size1 = 0;
    inFile.read(reinterpret_cast<char*>(&size1), sizeof(size1));
    vecVals.resize(size1);
    inFile.read(reinterpret_cast<char*>(vecVals.data()), size1 * sizeof(double));

    // Read columns vector
    size_t size2 = 0;
    inFile.read(reinterpret_cast<char*>(&size2), sizeof(size2));
    vecCols.resize(size2);
    inFile.read(reinterpret_cast<char*>(vecCols.data()), size2 * sizeof(int));

    // Read rows vector
    size_t size3 = 0;
    inFile.read(reinterpret_cast<char*>(&size3), sizeof(size3));
    vecRows.resize(size3);
    inFile.read(reinterpret_cast<char*>(vecRows.data()), size3 * sizeof(int));

    inFile.close();
}

void loadBCRSMatrixVectors(std::vector<double>& vecVals, std::vector<unsigned int>& vecCols, std::vector<unsigned int>& vecRows, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading." << std::endl;
        return;
    }

    // Read values vector
    size_t size1 = 0;
    inFile.read(reinterpret_cast<char*>(&size1), sizeof(size1));
    vecVals.resize(size1);
    inFile.read(reinterpret_cast<char*>(vecVals.data()), size1 * sizeof(double));

    // Read columns vector
    size_t size2 = 0;
    inFile.read(reinterpret_cast<char*>(&size2), sizeof(size2));
    vecCols.resize(size2);
    inFile.read(reinterpret_cast<char*>(vecCols.data()), size2 * sizeof(unsigned int));

    // Read rows vector
    size_t size3 = 0;
    inFile.read(reinterpret_cast<char*>(&size3), sizeof(size3));
    vecRows.resize(size3);
    inFile.read(reinterpret_cast<char*>(vecRows.data()), size3 * sizeof(unsigned int));

    inFile.close();
}

std::vector<double> loadResVector(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);  // Open file in binary mode
    if (!inFile) {
        std::cerr << "Error opening file for reading." << std::endl;
        return {};
    }

    // Read the size of the vector
    size_t size = 0;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Resize the vector to hold the elements
    std::vector<double> vec(size);

    // Read the contents of the vector
    inFile.read(reinterpret_cast<char*>(vec.data()), size * sizeof(double));

    inFile.close();
    return vec;
}
