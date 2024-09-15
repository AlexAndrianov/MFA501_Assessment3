#pragma once
#include <stdexcept>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_set>

namespace math {

template<class T>
class Vector {
public:
    Vector() = default;
    Vector(const Vector&) = default;
    Vector(std::vector<T> &&v) : _v(std::move(v)) {}

    Vector(size_t sz, const T &def_v)
    {
        _v.resize(sz, T{def_v});
    }

    T &operator[](size_t i) {
        return _v[i];
    }

    const T &operator[](size_t i) const {
        return _v[i];
    }

    friend std::ostream& operator<<(std::ostream& out, const Vector& vc)
    {
        for(auto i = 0u; i < vc.size(); i++)
        {
            if constexpr (std::is_same_v<T, int>) {
                out << (vc._v[i] == 0 ? " " : std::to_string(vc._v[i])) << " ";
            }
            else {
                out << vc._v[i] << " ";
            }
        }

        out << std::endl;
        return out;
    }

    size_t size() const { return _v.size(); }

    T summSquare() const {
        return std::accumulate(_v.begin() + 1, _v.end(), _v[0]*_v[0],
                               [](T total, const auto& item) {
                                   return total + item * item;
                               });
    }

    std::unordered_set<size_t> getRandomNIndexes(size_t size) const
    {
        static std::random_device rd;
        static std::mt19937 rng(rd());

        std::vector<size_t> indices(_v.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        indices.resize(size);
        return std::unordered_set<size_t>{indices.begin(), indices.end()};
    }

    Vector crossover(const Vector &v2, double diff) const
    {
        if constexpr (std::is_trivial_v<T>)
        {
            const auto indexesA = getRandomNIndexes(_v.size() * diff + 1);
            std::vector<T> res;

            for(auto i = 0u; i < _v.size(); i++)
            {
                res.push_back(indexesA.find(i) != indexesA.end() ? _v[i] : v2[i]);
            }

            return res;
        }
        else
        {
            Vector<T> res(_v.size(), _v[0]);
            for(auto i = 0u; i < _v.size(); i++)
                res._v[i] = _v[i].crossover(v2._v[i], diff);

            return res;
        }
    }

    std::vector<T> _v;
};

template <class T>
class Matrix {
public:
    Matrix() = default;
    Matrix(const Matrix&) = default;

    Matrix(size_t row, size_t col, const T &def_v)
    {
        if(!row || !col)
            throw std::runtime_error("Empty matrix");

        _v.resize(row, Vector<T>(col, def_v));
    }

    const Matrix &operator -= (const Matrix &mtrB) {
        auto &mtrA = *this;

        if(mtrA._v.size() != mtrB._v.size())
            throw std::runtime_error("Unconsistent matrices");

        if(mtrA._v[0].size() != mtrB._v[0].size())
            throw std::runtime_error("Unconsistent matrices");

        for(auto i = 0u; i < mtrA._v.size(); i++)
            for(auto j = 0u; j < mtrA._v[0].size(); j++)
                mtrA(i,j) -= mtrB(i,j);

        return mtrA;
    }

    Matrix operator - (const Matrix &mtrB) const {
        auto mtrA = *this;
        mtrA -= mtrB;
        return mtrA;
    }

    Matrix operator * (const double &v) const {
        const auto &mtrA = *this;

        Matrix res{mtrA._v.size(), mtrA._v[0].size(), 0.0};

        for(auto i = 0u; i < mtrA._v.size(); i++)
            for(auto j = 0u; j < mtrA._v[0].size(); j++)
                res(i, j) = mtrA(i, j) * v;

        return res;
    }

    bool isEmpty() const
    {
        return _v.empty();
    }

    bool isSquare() const
    {
        if(_v.empty())
            return true;

        return _v.size() == _v[0].size();
    }

    void addRow(Vector<T> &&row)
    {
        if(!_v.empty() && _v[0].size() != row.size())
            throw std::runtime_error("Numbers of columns is defferent");

        _v.emplace_back(std::move(row));
    }

    T &operator()(size_t r, size_t c) {
        return _v[r][c];
    }

    const T &operator()(size_t r, size_t c) const {
        return _v[r][c];
    }

    friend std::ostream& operator<<(std::ostream& out, const Matrix& mtx)
    {
        for(auto i = 0u; i < mtx._v.size(); i++)
            out << mtx._v[i];

        out << std::endl;
        return out;
    }

    T summSquare() const {
        return std::accumulate(_v.begin() + 1, _v.end(), _v[0].summSquare(),
                                  [](T total, const auto& item) {
                                      return total + item.summSquare();
                                  });
    }

    Matrix crossover(const Matrix &m2, double diff) const
    {
        Matrix<T> res{_v.size(), _v[0].size(), _v[0][0]};
        for(auto i = 0u; i < _v.size(); i++)
            res._v[i] = _v[i].crossover(m2._v[i], diff);

        return res;
    }

    std::vector<Vector<T>> _v;
};

}
