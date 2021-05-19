#include "CartPoleRenderer.hpp"

void Renderer::Create()
{
    glfwInit();
    window_ = glfwCreateWindow(680, 480, "CartPole", 0, 0);

    glfwMakeContextCurrent(window_);
}

void Renderer::Close()
{
    glfwDestroyWindow(window_);
    window_ = nullptr;

    glfwTerminate();
}

void Renderer::Render(const CartPole::State& state)
{
    glfwPollEvents();

    const float x = state.Desc.x;
    const float theta = state.Desc.theta;

    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-5, 5, -5, 5, -100, 100);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(x * 2, -3.f, 0);

    glPushMatrix();
    glScalef(1, 0.4f, 1);

    glColor3ub(255, 0, 0);
    DrawBox();
    glPopMatrix();

    glPushMatrix();

    glRotatef(theta * 57.29577951308232f, 0, 0, 1);
    glTranslatef(0, 1.8f, 0);
    glScalef(0.2f, 1.8f, 1);

    glColor3ub(0, 0, 255);
    DrawBox();

    glPopMatrix();

    glfwSwapBuffers(window_);
}

void Renderer::DrawBox() const
{
    glBegin(GL_QUADS);

    glVertex2f(-1, -1);
    glVertex2f(1, -1);
    glVertex2f(1, 1);
    glVertex2f(-1, 1);

    glEnd();
}
