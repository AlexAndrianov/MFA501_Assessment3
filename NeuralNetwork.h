#pragma once
#include "Matrix.h"
#include <math.h>

namespace math {

struct Neuron {
    Neuron(const Neuron&) = default;

    Neuron(size_t size)
        : _weights(size, size, 0.0)
    {
        for(auto i = 0u; i < _weights._v.size(); i++)
            for(auto j = 0u; j < _weights._v[0].size(); j++)
                _weights(i, j) = ((double)rand() / RAND_MAX) * 2 - 1;

        _b = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    double produce(const Matrix<double> &_input) const
    {
        double res = _b;

        for(auto i = 0u; i < _input._v.size(); i++)
            for(auto j = 0u; j < _input._v[0].size(); j++)
                res += _input(i, j) * _weights(i, j);

        return res;
    }

    Neuron crossover(const Neuron &n2, double diff) const
    {
        Neuron res(_weights._v.size());
        res._weights = _weights.crossover(n2._weights, diff);
        res._b = _b;
        return res;
    }

    Matrix<double> _weights;
    double _b = 0.0;
};

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
public:

    NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = default;

    NeuralNetwork(const Matrix<double> &inputLayer)
        : _inputLayer(inputLayer)
        , _outputLayer(inputLayer._v.size(), inputLayer._v.size(), inputLayer._v.size())
    {
        for(auto i = 0u; i < _inputLayer._v.size(); i++)
            for(auto j = 0u; j < _inputLayer._v[0].size(); j++)
                _outputLayer(i, j) = Neuron(inputLayer._v.size());
    }

    Matrix<double> forwardPass(bool der = false) const
    {
        Matrix<double> res{_inputLayer._v.size(), _inputLayer._v[0].size(), 0};

        for(auto i = 0u; i < _inputLayer._v.size(); i++)
        {
            for(auto j = 0u; j < _inputLayer._v[0].size(); j++)
            {
                const auto prod_res = _outputLayer(i, j).produce(_inputLayer);
                res(i, j) = der ? sigmoid_derivative(prod_res) : sigmoid(prod_res);
            }
        }

        return res;
    }

    Matrix<int> produceOutput() const
    {
        Matrix<double> produceRes = forwardPass();
        Matrix<int> res{produceRes._v.size(), produceRes ._v[0].size(), 0};

        for(auto i = 0u; i < produceRes._v.size(); i++)
            for(auto j = 0u; j < produceRes._v[0].size(); j++)
                res(i, j) = produceRes(i, j) < 0.5 ? 0 : 1;

        return res;
    }

    // fitness scores
    double meanSquareError(const Matrix<double> &ethalon) const
    {
        const auto diff = forwardPass(false) - ethalon;
        return diff.summSquare() / (ethalon._v.size() * ethalon._v[0].size());
    }

    void backPropogationLearn(const Matrix<double> &ethalon, double _learning_rate)
    {
        const double N = _outputLayer._v.size() * _outputLayer._v[0].size();
        const auto diff = forwardPass(false) - ethalon;
        const auto sigmoid_derivative_prod = forwardPass(true);

        Matrix<double> derevatives_matrix_b{_inputLayer._v.size(), _inputLayer._v[0].size(), 0.f};

        for(auto i = 0u; i < _inputLayer._v.size(); i++)
            for(auto j = 0u; j < _inputLayer._v[0].size(); j++)
                derevatives_matrix_b(i, j) = (diff(i, j) * sigmoid_derivative_prod(i, j) * 1.0) / N;

        Matrix<Matrix<double>> derevatives_matrix_w{_inputLayer._v.size(), _inputLayer._v[0].size(),
                                                    Matrix<double>{_inputLayer._v.size(), _inputLayer._v[0].size(), 0.f}};

        for(auto i = 0u; i < _inputLayer._v.size(); i++)
        {
            for(auto j = 0u; j < _inputLayer._v[0].size(); j++)
            {
                derevatives_matrix_w(i, j) = _inputLayer * diff(i, j) * sigmoid_derivative_prod(i, j) * (1.0 / N);
            }
        }

        // learning neurons
        for(auto i = 0u; i < _inputLayer._v.size(); i++)
        {
            for(auto j = 0u; j < _inputLayer._v[0].size(); j++)
            {
                _outputLayer(i, j)._b -= derevatives_matrix_b(i, j) * _learning_rate * -1.0;
                _outputLayer(i, j)._weights -= derevatives_matrix_w(i, j) * _learning_rate * -1.0;
            }
        }
    }

    NeuralNetwork crossover(const NeuralNetwork &n2, double diff) const
    {
        NeuralNetwork res(_inputLayer);
        res._outputLayer = _outputLayer.crossover(n2._outputLayer, diff);
        return res;
    }

    Matrix<double> _inputLayer;
    Matrix<Neuron> _outputLayer;
};

}
