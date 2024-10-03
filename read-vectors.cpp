
#include "read-vectors.hpp"

template <typename I>
void loadSparseMatrixVectors(std::vector<double>& vecVals, std::vector<I>& vecCols, std::vector<I>& vecRows, const std::string& filename)
{
    char absPath[PATH_MAX];
    if (realpath(filename.c_str(), absPath) == nullptr) {
        std::cerr << "Error resolving absolute path: " << filename << std::endl;
        return;
    }

    std::ifstream inFile(absPath, std::ios::in | std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file: " << std::strerror(errno)<< std::endl;
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
    inFile.read(reinterpret_cast<char*>(vecCols.data()), size2 * sizeof(I));

    // Read rows vector
    size_t size3 = 0;
    inFile.read(reinterpret_cast<char*>(&size3), sizeof(size3));
    vecRows.resize(size3);
    inFile.read(reinterpret_cast<char*>(vecRows.data()), size3 * sizeof(I));

    inFile.close();
}

template void loadSparseMatrixVectors(std::vector<double>&, std::vector<int>&, std::vector<int>&, const std::string&);
template void loadSparseMatrixVectors(std::vector<double>&, std::vector<unsigned int>&, std::vector<unsigned int>&, const std::string&);

std::vector<double> loadResVector(const std::string& filename)
{
    char absPath[PATH_MAX];
    if (realpath(filename.c_str(), absPath) == nullptr) {
        std::cerr << "Error resolving absolute path: " << filename << std::endl;
    }

    std::ifstream inFile(absPath, std::ios::in | std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file: " << std::strerror(errno)<< std::endl;
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

void squareCSCtoCSR(std::vector<double> Dvals, std::vector<int> Drows, std::vector<int> Dcols, std::vector<double>& Dvals_, std::vector<int>& Drows_, std::vector<int>& Dcols_)
{
    unsigned int sizeDvals = size(Dvals);
    unsigned int sizeDcols = size(Dcols);
    unsigned int sizeDrows = size(Drows);

    std::vector<int> Cols(sizeDvals);

    for(int i=0; i<sizeDcols-1; i++){
      for(int j=Dcols[i];j<Dcols[i+1];j++){
        Cols[j] = i;
      }
    }
    //std::cout << size(Cols) << std::endl;
    //for(const auto& val : Cols) std::cout << val << " ";
    //std::cout << std::endl;

    std::vector<std::tuple<int,double,int>> ConvertVec;

    for(int i=0; i<sizeDvals; i++){
      ConvertVec.push_back(std::make_tuple(Drows[i],Dvals[i],Cols[i]));
    }

    //std::cout << size(ConvertVec) << std::endl;
    //std::cout << "############ Before sorting ############" << std::endl;
    //for (const auto& val : ConvertVec) std::cout << "{" << std::get<0>(val) << ", "  << std::get<1>(val) << ", " << std::get<2>(val) << "}" << " ";
    //std::cout << std::endl;

    std::sort(ConvertVec.begin(),ConvertVec.end());

    auto it = ConvertVec.begin();
    while (it != ConvertVec.end()) {
        auto range_end = std::find_if(it, ConvertVec.end(),
            [it](const std::tuple<int, double, int>& tup) {
                return std::get<0>(tup) != std::get<0>(*it);
            });


        std::sort(it, range_end,
            [](const std::tuple<int, double, int>& a, const std::tuple<int, double, int>& b) {
                return std::get<2>(a) < std::get<2>(b);
            });

        it = range_end;
    }

    //std::cout << "############ After sorting ############" << std::endl;
    //for (const auto& val : ConvertVec) std::cout << "{" << std::get<0>(val) << ", "  << std::get<1>(val) << ", " << std::get<2>(val) << "}" << " ";
    //std::cout << std::endl;

    for(int i=0; i<sizeDvals; i++){
        //std::cout << "{" << std::get<0>(ConvertVec[i]) << ", "  << std::get<1>(ConvertVec[i]) << ", " << std::get<2>(ConvertVec[i]) << "}" << std::endl;
        Dvals_[i] = std::get<1>(ConvertVec[i]);
        Dcols_[i] = std::get<2>(ConvertVec[i]);
    }

    for(int target = 0; target< sizeDcols-1; target++){
      auto it = std::find_if(ConvertVec.begin(), ConvertVec.end(),
          [target](const std::tuple<int, double, int>& tup) {
              return std::get<0>(tup) == target;
          });

      if (it != ConvertVec.end()) {
          Drows_[target] = std::distance(ConvertVec.begin(),it);
      } else {
          std::cout << target << " not found in the third element of any tuple.\n";
      }
    }

    for (int i=1; i<sizeDcols-1; i++){
        if(Drows_[i] == 0){
            Drows_[i] = Drows_[i+1];
        }
    }
    Drows_[sizeDcols-1] = Dcols[sizeDcols-1];
}

double* squareCSCtoMatrix(std::vector<double> Dvals, std::vector<int> Drows, std::vector<int> Dcols)
{
    int lda = size(Dcols)-1;
    int nnzs = size(Dvals);

    double* Dmatrix = (double*)malloc(sizeof(double)*lda*lda);

    std::vector<int> Cols(nnzs);

    for(int i=0; i<lda; i++){
      for(int j=Dcols[i];j<Dcols[i+1];j++){
        Cols[j] = i;
      }
    }

    for(int i=0; i<(lda*lda); i++){
        Dmatrix[i] = 0;
    }

    for(int i=0; i<nnzs; i++){
        Dmatrix[Drows[i]+Cols[i]*lda] = Dvals[i];
    }

    return Dmatrix;
}
