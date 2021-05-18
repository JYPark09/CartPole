#pragma once

#include <random>

struct Random final
{
    static std::mt19937& Engine()
    {
        static std::mt19937 engine(12345);

        return engine;
    }
};
