import typing as type
from point import Point


class TrapezoidFunction:
    _pointLeftDown: Point
    _pointLeftTop: Point
    _pointRightTop: Point
    _pointRightDown: Point
    _leftSharp: bool = False
    _rightSharp: bool = False

    def __init__(self, pointLeftDown: Point, pointLeftTop: Point, pointRightTop: Point, pointRightDown: Point) -> None:
        self._pointLeftDown = pointLeftDown
        self._pointLeftTop = pointLeftTop
        self._pointRightTop = pointRightTop
        self._pointRightDown = pointRightDown
        if self._pointLeftDown.x == self._pointLeftTop.x:
            self._leftSharp = True
        if self._pointRightDown.x == self._pointRightTop.x:
            self._rightSharp = True

    def value(self, x: float) -> float:
        if x <= self._pointLeftDown.x or x >= self._pointRightDown.x:
            return 0
        if x > self._pointLeftDown.x and x < self._pointLeftTop.x and not self._leftSharp:
            return 1 * (x - self._pointLeftDown.x)/(self._pointLeftTop.x - self._pointLeftDown.x)
        elif x > self._pointRightTop.x and x < self._pointRightDown.x and not self._rightSharp:
            return 1 * (self._pointRightDown.x - x)/(self._pointRightDown.x - self._pointRightTop.x)
        elif self._leftSharp and x < self._pointLeftTop.x:
            return 0
        elif self._rightSharp and x > self._pointRightTop.x:
            return 0
        else:
            return 1
