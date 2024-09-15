#pragma once
#include "NeuralNetwork.h"
#include <algorithm>

namespace math {

class Population {
public:
    Population(int size, double learning_rate, const math::Matrix<double> &input, const math::Matrix<double> &output)
        : _learning_rate(learning_rate)
        , _number_of_best(size * 0.1)
        , _ethalonOutput(output)
    {
        for(auto i = 0u; i < size; i++)
        {
            _population.emplace_back(input);
        }
    }

    // little changing of neurons weights
    void mutate()
    {
        for(auto &nn : _population)
            nn.backPropogationLearn(_ethalonOutput, _learning_rate);
    }

    void selection() {
        std::sort(_population.begin(), _population.end(), [this](const auto& a, const auto& b) {
            return a.meanSquareError(_ethalonOutput) < b.meanSquareError(_ethalonOutput);
        });
    }

    NeuralNetwork crossover(const NeuralNetwork &parent1, const NeuralNetwork& parent2)
    {
        const auto mse1 = parent1.meanSquareError(_ethalonOutput);
        const auto mse2 = parent2.meanSquareError(_ethalonOutput);

         // number wheights from the network with smaller mse (in percentage relation)
        const auto diff = [](auto littleV, auto biggerV)
        {
            return 1.0 - (littleV / (2.0 * biggerV));
        };

        if(mse1 <= mse2)
            return parent1.crossover(parent2, diff(mse1, mse2));
        else
            return parent2.crossover(parent1, diff(mse2, mse1));
    }

    void geneticIteration()
    {
        selection();

        const size_t size = _population.size();
        _population.resize(_number_of_best + 1);

        auto i = 0u;
        while(true)
        {
            _population.push_back(crossover(_population[i], _population[i + 1]));
            i++;

            if(_population.size() >= size)
                break;
        }

        mutate();
    }

    const NeuralNetwork &leader() const { return _population.front(); }

    double _learning_rate;
    int _number_of_best;
    std::vector<NeuralNetwork> _population;
    math::Matrix<double> _ethalonOutput;
};

}
