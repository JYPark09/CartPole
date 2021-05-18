#include <iostream>

#include "CartPole.hpp"
#include "CartPoleRenderer.hpp"

using std::cout;
using std::endl;

int main()
{
    Renderer renderer;
    renderer.Create();

    CartPole::Env env;

    CartPole::State state = env.Reset();
    while (true)
    {
        renderer.Render(state);

        auto [newState, reward, done] = env.Step(CartPole::Action::RIGHT);

        cout << "x: " << state.Desc.x << " d_x: " << state.Desc.d_x
             << " theta: " << state.Desc.theta
             << " d_theta: " << state.Desc.d_theta << " reward: " << reward
             << endl;

        state = newState;

        if (done)
            break;
    }
    while (true)
        ;

    renderer.Close();
}
