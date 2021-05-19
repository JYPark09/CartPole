#pragma once

#include <Windows.h>
#include <tiny_dnn/tiny_dnn.h>
#include <vector>
#include <tuple>

#include "CartPole.hpp"

class Agent final
{
 public:
    Agent();

    std::tuple<CartPole::Action, tiny_dnn::vec_t> GetAction(const CartPole::State& state);

    void Clear();
    void AddInterActionInfo(const CartPole::State& state, CartPole::Action action, tiny_dnn::vec_t pred, float reward);

    void Train();

 private:
    tiny_dnn::network<tiny_dnn::sequential> net_;
    tiny_dnn::adam opt_;

    std::vector<tiny_dnn::vec_t> states_;
    std::vector<CartPole::Action> actions_;
    std::vector<tiny_dnn::vec_t> preds_;
    std::vector<float> rewards_;
};
