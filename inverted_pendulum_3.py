from PyQt5.QtCore import forcepoint
import numpy as np
from numpy import pi, sin, cos, arctan2
from itertools import cycle
from sys import argv, exit
from numpy import typing
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
import typing as type
import time
import matplotlib.pyplot as plt
from point import Point
from trapezoidFunction import TrapezoidFunction

from triangleFunction import TriangleFunction


class InvertedPendulum(QtGui.QWidget):
    '''Inicjalizacja stałych:
    M - masa wózka
    m - masa kulki
    l - długość ramienia wahadła

    Warunków początkowych:
    x0 - początkowe położenie wózka
    dx0 - początkowa prędkość wózka
    theta0 - początkowe położenie wahadla
    dtheta0 - początkowa prędkość wahadła

    Zakłócenia zewnętrznego:
    dis_cyc - zmienna odpowiada za to, czy zakłócenie jest zapętlone
    disruption - wartości zakłócenia w kolejnych chwilach czasowych

    Parametry planszy/obrazka:
    iw, ih - szerokość i wysokość obrazka
    x_max - maksymalna współrzędna pozioma (oś x jest symetryczna, więc minimalna wynosi -x_max)
    h_min - minialna współrzędna pionowa
    h_max - maksymalna współrzędna pionowa

    Powyższe dane są pobierane z pliku jeśli zmienna f_name nie jest pusta'''

    def __init__(self, M=10, m=5, l=50, x0=0, theta0=0, dx0=0, dtheta0=0, dis_cyc=True, disruption=[0], iw=1000, ih=500, x_max=100, h_min=0, h_max=100, f_name=None):
        if f_name:
            with open(f_name) as f_handle:
                lines = f_handle.readlines()
                init_cond = lines[0].split(' ')
                self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = [
                    float(el) for el in init_cond[:7]]
                self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = [
                    int(el) for el in init_cond[-5:]]
                if lines[1]:
                    self.disruption = cycle([float(el)
                                            for el in lines[2].split(' ')])
                else:
                    self.disruption = iter([float(el)
                                           for el in lines[2].split(' ')])
        else:
            self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = M, m, l, x0, theta0, dx0, dtheta0
            self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = iw, ih, x_max, h_min, h_max
            if dis_cyc:
                self.disruption = cycle(disruption)
            else:
                self.disruption = iter(disruption)
        super(InvertedPendulum, self).__init__(parent=None)

    # Inicjalizacja obrazka
    def init_image(self):
        self.h_scale = self.image_h/(self.h_max-self.h_min)
        self.x_scale = self.image_w/(2*self.x_max)
        self.hor = (self.h_max-10)*self.h_scale
        self.c_w = 16*self.x_scale
        self.c_h = 8*self.h_scale
        self.r = 8
        self.x = self.x0
        self.theta = self.theta0
        self.dx = self.dx0
        self.dtheta = self.dtheta0
        self.setFixedSize(self.image_w, self.image_h)
        self.show()
        self.setWindowTitle("Inverted Pendulum")
        self.update()

    # Rysowanie wahadła i miarki
    def paintEvent(self, e):
        x, x_max, x_scale, theta = self.x, self.x_max, self.x_scale, self.theta
        hor, l, h_scale = self.hor, self.l, self.h_scale
        image_w, c_w, c_h, r, image_h, h_max, h_min = self.image_w, self.c_w, self.c_h, self.r, self.image_h, self.h_max, self.h_min
        painter = QtGui.QPainter(self)
        painter.setPen(pg.mkPen('k', width=2.0*self.h_scale))
        painter.drawLine(0, hor, image_w, hor)
        painter.setPen(pg.mkPen((165, 42, 42), width=2.0*self.x_scale))
        painter.drawLine(x_scale*(x+x_max), hor, x_scale *
                         (x+x_max-l*sin(theta)), hor-h_scale*(l*cos(theta)))
        painter.setPen(pg.mkPen('b'))
        painter.setBrush(pg.mkBrush('b'))
        painter.drawRect(x_scale*(x+x_max)-c_w/2, hor-c_h/2, c_w, c_h)
        painter.setPen(pg.mkPen('r'))
        painter.setBrush(pg.mkBrush('r'))
        painter.drawEllipse(x_scale*(x+x_max-l*sin(theta)-r/2),
                            hor-h_scale*(l*cos(theta)+r/2), r*x_scale, r*h_scale)
        painter.setPen(pg.mkPen('k'))
        for i in np.arange(-x_max, x_max, x_max/10):
            painter.drawText((i+x_max)*x_scale, image_h-10, str(int(i)))
        for i in np.arange(h_min, h_max, (h_max-h_min)/10):
            painter.drawText(0, image_h-(int(i)-h_min)*h_scale, str(int(i)))

    # Rozwiązanie równań mechaniki wahadła
    def solve_equation(self, F):
        l, m, M = self.l, self.m, self.M
        g = 9.81
        a11 = M+m
        a12 = -m*l*cos(self.theta)
        b1 = F-m*l*self.dtheta**2*sin(self.theta)
        a21 = -cos(self.theta)
        a22 = l
        b2 = g*sin(self.theta)
        a = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        sol = np.linalg.solve(a, b)
        return sol[0], sol[1]

    # Scałkowanie numeryczne przyśpieszenia, żeby uzyskać pozostałe parametry układu
    def count_state_params(self, F, dt=0.001):
        ddx, ddtheta = self.solve_equation(F)
        self.dx += ddx*dt
        self.x += self.dx*dt
        self.dtheta += ddtheta*dt
        self.theta += self.dtheta*dt
        self.theta = arctan2(sin(self.theta), cos(self.theta))

    # Uruchomienie symulacji
    # Zmienna sandbox mówi o tym, czy symulacja ma zostać przerwana w przypadku nieudanego sterowania -
    # - to znaczy takiego, które pozwoliło na zbyt duże wychylenia iksa lub na zbyt poziomo położenie wahadła
    def run(self, sandbox, frameskip=20):
        self.sandbox = sandbox
        self.frameskip = frameskip
        self.init_image()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.single_loop_run)
        timer.start(1)

    # n - krotne obliczenie następnego stanu układu
    # Gdzie n - to frameskip
    def single_loop_run(self):
        for i in range(self.frameskip+1):
            dis = next(self.disruption, 0)
            control = self.fuzzy_control(
                self.x, self.theta, self.dx, self.dtheta)
            F = dis+control
            self.count_state_params(F)
            if not self.sandbox:
                if self.x < -self.x_max or self.x > self.x_max or np.abs(self.theta) > np.pi/3:
                    exit(1)
        self.update()

    # Regulator rozmyty, który trzeba zaimplementować
    def fuzzy_control(self, x, theta, dx, dtheta):

        # reguły utrzymujące wahadło w pionie z gasnącymi oscylacjami
        doNothing: float = self.fuzzy_and(self.pendulum_in_center.value(
            theta), self.pendulum_zero_speed.value(dtheta))

        goLeft: float = self.fuzzy_and(self.pendulum_slightly_left.value(theta), self.fuzzy_or(
            self.pendulum_slow_left.value(theta), self.pendulum_fast_left.value(theta)))
        goRight: float = self.fuzzy_and(self.pendulum_slightly_right.value(theta), self.fuzzy_or(
            self.pendulum_slow_right.value(theta), self.pendulum_fast_right.value(theta)))

        goStrongLeft: float = self.fuzzy_and(self.pendulum_left.value(theta), self.fuzzy_or(
            self.pendulum_slow_left.value(theta), self.pendulum_fast_left.value(theta)))
        goStrongRight: float = self.fuzzy_and(self.pendulum_right.value(theta), self.fuzzy_or(
            self.pendulum_slow_right.value(theta), self.pendulum_fast_right.value(theta)))

        pushLightLeft: float = self.fuzzy_and(self.fuzzy_not(self.pendulum_in_center.value(
            theta)), self.pendulum_slow_left.value(dtheta))

        pushLightRight: float = self.fuzzy_and(self.fuzzy_not(self.pendulum_in_center.value(
            theta)), self.pendulum_slow_right.value(dtheta))

        #cart_nothing, cart_light_left, cart_light_right, cart_midium_left, cart_midium_right, cart_strong_left, cart_strong_right
        weights: type.List[float] = [doNothing, pushLightLeft,
                                     pushLightRight,  goLeft, goRight, goStrongLeft, goStrongRight]

        regulator_output: float = self.calc_regulator_output(
            self.cart_force_values, weights)

        return regulator_output

    #############################
    # ADDED FUNCTIONS/VARIABLES #
    #############################

    # angular position
    pendulum_in_center = TriangleFunction(
        Point(np.deg2rad(-1), 0), Point(0, 1), Point(np.deg2rad(1), 0))
    pendulum_slightly_left = TriangleFunction(
        Point(np.deg2rad(0), 0), Point(np.deg2rad(2), 1), Point(np.deg2rad(4), 0))
    pendulum_slightly_right = TriangleFunction(
        Point(np.deg2rad(-4), 0), Point(np.deg2rad(-2), 1), Point(np.deg2rad(0), 0))
    pendulum_left = TrapezoidFunction(Point(np.deg2rad(3), 0), Point(
        np.deg2rad(8), 1), Point(np.deg2rad(13), 1), Point(np.deg2rad(13), 0))
    pendulum_right = TrapezoidFunction(Point(np.deg2rad(-13), 0), Point(
        np.deg2rad(-13), 1), Point(np.deg2rad(-8), 1), Point(np.deg2rad(-3), 0))

    # angular speed
    pendulum_zero_speed = TriangleFunction(
        Point(-0.005, 0), Point(0, 1), Point(0.005, 0))
    pendulum_slow_left = TriangleFunction(
        Point(0, 0), Point(0.006, 1), Point(0.01, 0))
    pendulum_slow_right = TriangleFunction(
        Point(-0.01, 0), Point(-0.006, 1), Point(0, 0))
    pendulum_fast_left = TrapezoidFunction(
        Point(0.009, 0), Point(0.014, 1), Point(1, 1), Point(1, 0))
    pendulum_fast_right = TrapezoidFunction(
        Point(-1, 0), Point(-1, 1), Point(-0.014, 1), Point(-0.009, 0))

    # cart position
    cart_in_zero = TriangleFunction(
        Point(-1, 0), Point(0, 1), Point(1, 0))
    cart_on_left = TriangleFunction(
        Point(-40, 0), Point(-40, 1), Point(-0.8, 0))
    cart_on_right = TriangleFunction(
        Point(0.8, 0), Point(40, 1), Point(40, 0))

    # cart speed
    cart_zero_speed = TriangleFunction(
        Point(-0.5, 0), Point(0, 1), Point(0.5, 0))
    cart_slowly_right = TriangleFunction(
        Point(0.3, 0), Point(0.5, 1), Point(0.11, 0))
    cart_slowly_left = TriangleFunction(
        Point(-0.11, 0), Point(-0.5, 1), Point(-0.3, 0))
    cart_fast_right = TrapezoidFunction(
        Point(0.1, 0), Point(0.2, 1), Point(2, 1), Point(2, 0)
    )
    cart_fast_left = TrapezoidFunction(
        Point(-2, 0), Point(-2, 1), Point(-0.2, 1), Point(-0.1, 0)
    )

    cart_nothing: float = 0
    cart_light_left: float = -5
    cart_light_right: float = 5
    cart_midium_left: float = -50
    cart_midium_right: float = 50
    cart_strong_left: float = -100
    cart_strong_right: float = 100
    cart_force_values: type.List[float] = [cart_nothing, cart_light_left, cart_light_right,
                                           cart_midium_left, cart_midium_right, cart_strong_left, cart_strong_right]

    def fuzzy_not(self, value: float) -> float:
        return 1 - value

    def fuzzy_and(self, *args: float) -> float:
        return min(args)

    def fuzzy_or(self, *args: float) -> float:
        return max(args)

    def calc_regulator_output(self, values: type.List[float], weights: type.List[float]) -> float:
        output: float = 0
        try:
            output = np.average(values, axis=0, weights=weights)
        except ZeroDivisionError:
            return 0

        return output

    def cut_off_value(self, value: float, treshold: float) -> float:
        if value > treshold:
            return treshold
        else:
            return value


if __name__ == '__main__':
    app = QtGui.QApplication(argv)
    if len(argv) > 1:
        ip = InvertedPendulum(f_name=argv[1])
    else:
        ip = InvertedPendulum(x0=90, dx0=0, theta0=0,
                              dtheta0=0.1, ih=800, iw=1000, h_min=-80, h_max=80)
    ip.run(sandbox=False)
    exit(app.exec_())
