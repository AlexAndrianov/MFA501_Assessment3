#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Population.h"

using namespace std;

template<typename T>
math::Matrix<T> readMatrix(std::ifstream &inputFile)
{
    math::Matrix<T> mtx;

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream lineStream(line);

        std::vector<T> values;
        double value;

        while (lineStream >> value) {
            values.push_back(value);
        }

        mtx.addRow(math::Vector(std::move(values)));
    }

    return mtx;
}

int main(int argc, char* argv[])
{
    if(argc < 2 || !argv)
    {
        cout << "Enter the type of learning!" << endl;
        return -1;
    }

    const bool is_genetics = std::string(argv[1]) == "1";
    int number_of_population = 100;
    if(is_genetics)
    {
        if(argc >= 3)
        {
            number_of_population = std::atoi(argv[2]);
        }

        cout << "Size of population is : " << number_of_population << endl;
    }

    std::ifstream inputFile("matrix_input.txt");
    std::ifstream ethalonFile("matrix_ethalon.txt");

    if (!inputFile.is_open() || !ethalonFile.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    auto inputMatrixInt = readMatrix<int>(inputFile);

    // to prevent fake learning
    math::Matrix<double> inputMatrix{inputMatrixInt._v.size(), inputMatrixInt._v[0].size(), 0.};

    for(auto i = 0u; i < inputMatrixInt._v.size(); i++)
        for(auto j = 0u; j < inputMatrixInt._v[0].size(); j++)
            inputMatrix(i, j) = inputMatrixInt(i, j) == 1 ? 1.0 : -1.0;

    auto ethalonMatrix = readMatrix<double>(ethalonFile);

    if(inputMatrix.isEmpty())
    {
        std::cerr << "Matrix should not be empty" << std::endl;
        return 1;
    }

    if(!ethalonMatrix.isSquare())
    {
        std::cerr << "Matrix should be square" << std::endl;
        return 1;
    }

    if(is_genetics)
    {
        // the genetic algorithm approach
        math::Population population(number_of_population, 1, inputMatrix, ethalonMatrix);

        std::cout << "Genetic algorithm approach" << std::endl;
        std::cout << "Inputed matrix: " << std::endl;
        std::cout << inputMatrixInt;

        auto number_of_iteration = 0;

        while(true) {
            std::cout << "After " + std::to_string(number_of_iteration) + " iteration of algorithm to minimize " << std::endl;
            std::cout << "the objective function, the reconstructed image is:" << std::endl;
            std::cout << population.leader().produceOutput();

            std::cout << "Do you want to make next iteration?" << std::endl;
            std::cout << "Please enter y or n" << std::endl;

            std::string to_next;
            std::cin >> to_next;

            if(to_next != "y")
                return 0;

            number_of_iteration++;
            population.geneticIteration();

            std::cout << "Mean Square Error: " << population.leader().meanSquareError(ethalonMatrix);
        }
    }
    else
    {
        // the classic gradient descending
        std::cout << "Gradient descending approach" << std::endl;
        std::cout << "Inputed matrix: " << std::endl;
        std::cout << inputMatrixInt;

        math::NeuralNetwork network(inputMatrix);

        auto number_of_iteration = 0;
        while(true) {
            std::cout << "After " + std::to_string(number_of_iteration) + " iteration of algorithm to minimize " << std::endl;
            std::cout << "the objective function, the reconstructed image is:" << std::endl;
            std::cout << network.produceOutput();

            std::cout << "Do you want to make next 1 iterations?" << std::endl;
            std::cout << "Please enter y or n" << std::endl;

            std::string to_next;
            std::cin >> to_next;

            if(to_next != "y")
                return 0;

            number_of_iteration += 1;
            for (auto i = 0; i < 1; i++)
                network.backPropogationLearn(ethalonMatrix, 1);

            std::cout << "Mean Square Error: " << network.meanSquareError(ethalonMatrix);
        }
    }

    return 0;
}
