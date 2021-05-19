#include <deque>
#include <iostream>

#include "Agent.hpp"
#include "CartPole.hpp"
#include "CartPoleRenderer.hpp"

using std::cout;
using std::endl;

struct EpisodeResult final
{
    int steps{ 0 };
    float reward{ 0 };
};

//! Returns total reward
EpisodeResult ProcEpisode(Agent& agent, Renderer& renderer, bool render);

int main()
{
    Agent agent;
    Renderer renderer;
    renderer.Create();

    std::deque<float> results;

    agent.SetGamma(0.99f);
    agent.SetLearningRate(1e-3f);

    for (int ep = 1; ; ++ep)
    {
        const auto result = ProcEpisode(agent, renderer, (ep % 10 == 0));

        cout << "Episode " << ep << " - total reward: " << result.reward << endl;
    }

    renderer.Close();
}

EpisodeResult ProcEpisode(Agent& agent, Renderer& renderer, bool render)
{
    EpisodeResult result;

    agent.Clear();

    CartPole::Env env;

    CartPole::State state = env.Reset();
    for (result.steps = 0; result.steps < 500; ++result.steps)
    {
        if (render)
            renderer.Render(state);

        CartPole::Action action;
        tiny_dnn::vec_t pred;
        std::tie(action, pred) = agent.GetAction(state);

        CartPole::State newState;
        float reward;
        bool done;
        std::tie(newState, reward, done) = env.Step(action);
        result.reward += reward;

        if (done)
            reward = -100;

        agent.AddInterActionInfo(state, action, pred, reward);

        state = newState;

        if (done)
            break;
    }

    agent.Train();

    return result;
}
