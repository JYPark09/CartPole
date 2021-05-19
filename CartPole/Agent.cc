#include "Agent.hpp"

#include <iostream>
#include <random>

#include "Random.hpp"

namespace
{
constexpr float GAMMA = 0.99f;
}

Agent::Agent()
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using namespace tiny_dnn::layers;

    net_ << fc(4, 64) << relu() << fc(64, 64) << relu() << fc(64, 2)
         << softmax();
}

std::tuple<CartPole::Action, tiny_dnn::vec_t> Agent::GetAction(const CartPole::State& state)
{
    const tiny_dnn::vec_t input(state.Arr, state.Arr + 4);
    const auto pred = net_.predict(input);

    std::uniform_real_distribution<float> dist(0, 1);

    if (dist(Random::Engine()) > pred[0])
    {
        return { CartPole::Action::RIGHT, pred };
    }

    return { CartPole::Action::LEFT, pred };
}

void Agent::Clear()
{
    states_.clear();
    actions_.clear();
    preds_.clear();
    rewards_.clear();
}

void Agent::AddInterActionInfo(const CartPole::State& state, CartPole::Action action, tiny_dnn::vec_t pred, float reward)
{
    states_.emplace_back(state.Arr, state.Arr + 4);
    actions_.emplace_back(action);
    preds_.emplace_back(std::move(pred));
    rewards_.emplace_back(reward);
}

void Agent::Train()
{
    using namespace tiny_dnn;

    const int epLength = static_cast<int>(rewards_.size());

    // build returns
    float G = 0;
    std::vector<float> returns(epLength);
    for (int i = epLength - 1; i > 0; --i)
    {
        G = rewards_[i] + GAMMA * G;
        returns[i] = G;
    }

    // normalize
    const float mean = std::accumulate(begin(returns), end(returns), 0.f, [epLength](float a, float b) { return a + b / epLength; });
    float stddev = 0;
    for (auto ret : returns)
        stddev += ret * ret;
    stddev = std::sqrtf(stddev / epLength);

    for (auto& ret : returns)
    {
        ret = (ret - mean) / stddev;
    }

    tensor_t input(epLength), output(epLength);
    for (int i = 0; i < epLength; ++i)
    {
        input[i] = states_[i];
        output[i] = ((actions_[i] == CartPole::Action::LEFT) ? tiny_dnn::vec_t{ returns[i], 0 } : tiny_dnn::vec_t{ 0, returns[i] });
    }
    net_.train<mse, adam>(opt_, input, output, 1, 1);
}
