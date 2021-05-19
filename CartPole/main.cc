#include <deque>
#include <iostream>

#include "Agent.hpp"
#include "CartPole.hpp"
#include "CartPoleRenderer.hpp"

using std::cin;
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

    float gamma, lr;
    cout << "Input Gamma: ";
    cin >> gamma;
    cin.clear();

    cout << "Input Learning Rate: ";
    cin >> lr;
    cin.clear();

    agent.SetGamma(gamma);
    agent.SetLearningRate(lr);

    int ep;
    for (ep = 1; ; ++ep)
    {
        const auto result = ProcEpisode(agent, renderer, (ep % 50 == 0));

        cout << "Episode " << ep << " - total reward: " << result.reward << endl;
        results.emplace_back(result.reward);

        const int n = std::min(ep, 50);
        const int startIdx = std::max(ep - n, 0);
        const float meanReward =
            std::accumulate(begin(results) + startIdx, end(results), 0.f) / n;
        if (meanReward >= 450.f)
            break;
    }

    renderer.Close();

    cout << endl << "<training finished>" << endl << "total episodes: " << ep << endl;
    cout << "gamma: " << gamma << endl << "learning rate: " << lr << endl;
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
