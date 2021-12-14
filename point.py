import typing as type


class Point:
    x: float
    y: float
    position: type.Tuple[float, float]

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.position = (x, y)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Point):
            return NotImplemented

        return self.x == o.x and self.y == o.y

    def __repr__(self) -> str:
        return 'Point(x: %s, y: %s)' % (self.x, self.y)
