#pragma once

#include <Windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <GLFW/glfw3.h>

#include "CartPole.hpp"

class Renderer final
{
 public:
    void Create();
    void Close();

    void Render(const CartPole::State& state);

 private:
    void DrawBox() const;

 private:
    GLFWwindow* window_{ nullptr };
};
