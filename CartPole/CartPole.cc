// this code come from
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

#include "CartPole.hpp"

#include <cmath>
#include <cstring>
#include <random>

namespace
{
constexpr float GRAVITY = 9.8f;
constexpr float MASS_CART = 1.f;
constexpr float MASS_POLE = 0.1f;
constexpr float POLE_LENGTH = 0.5f;
constexpr float FORCE_MAG = 10.f;
constexpr float TAU = 0.02f;

constexpr float THETA_THRESHOLD =
    0.20943951023931953f;  // 12.f * 2.f * PI / 360.f;
constexpr float X_THRESHOLD = 2.4f;
}  // namespace

namespace CartPole
{
State Env::Reset()
{
    std::mt19937 engine(12345);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    for (int i = 0; i < 4; ++i)
        state_.Arr[i] = dist(engine);

    done_ = false;

    return state_;
}

auto Env::Step(Action action) -> std::tuple<State, Reward, bool>
{
    const auto [x, d_x, theta, d_theta] = state_.Desc;
    const float force = FORCE_MAG * (action == Action::RIGHT ? 1 : -1);

    const float sint = std::sin(theta);
    const float cost = std::cos(theta);

    const float temp =
        (force + MASS_POLE * POLE_LENGTH * (d_theta * d_theta) * sint) /
        (MASS_POLE + MASS_CART);
    const float theta_acc =
        (GRAVITY * sint - cost * temp) /
        (POLE_LENGTH *
         (4.f / 3.f - MASS_POLE * (cost * cost) / (MASS_POLE + MASS_CART)));
    const float x_acc = temp - (MASS_POLE * POLE_LENGTH) * theta_acc * cost /
                                   (MASS_POLE + MASS_CART);

    state_.Desc.x = x + TAU * d_x;
    state_.Desc.d_x = d_x + TAU * x_acc;
    state_.Desc.theta = theta + TAU * d_theta;
    state_.Desc.d_theta = d_theta + TAU * theta_acc;

    done_ = CheckDone();

    const float reward = !done_;

    return { state_, reward, done_ };
}

bool Env::CheckDone() const
{
    const auto [x, d_x, theta, d_theta] = state_.Desc;

    return (x < -X_THRESHOLD) || (x > X_THRESHOLD) ||
           (theta < -THETA_THRESHOLD) || (theta > THETA_THRESHOLD);
}
}  // namespace CartPole
