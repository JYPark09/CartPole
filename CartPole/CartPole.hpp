#pragma once

#include <tuple>

namespace CartPole
{
union State
{
    struct
    {
        float x;
        float d_x;
        float theta;
        float d_theta;
    } Desc;

    float Arr[4];
};

enum class Action
{
    LEFT,
    RIGHT
};

using Reward = float;

class Env
{
 public:
    State Reset();
    auto Step(Action action) -> std::tuple<State, Reward, bool>;
    
 private:
    bool CheckDone() const;

 private:
    bool done_{ false };
    State state_;
};
}  // namespace CartPole
