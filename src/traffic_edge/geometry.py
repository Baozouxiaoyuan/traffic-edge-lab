from __future__ import annotations


def validate_polygon(points: list) -> None:
    """Reject collapsed or self-intersecting lane geometry before analysis."""
    if len(points) < 3 or len(set(tuple(p) for p in points)) != len(points):
        raise ValueError("Draw at least three distinct polygon corners.")
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    if abs(area) < 1e-8:
        raise ValueError("The polygon has no area. Draw corners around a road region.")
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    for i in range(len(points)):
        a, b = points[i], points[(i+1) % len(points)]
        for j in range(i+1, len(points)):
            if j == i+1 or (i == 0 and j == len(points)-1):
                continue
            c, d = points[j], points[(j+1) % len(points)]
            if cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0:
                if (max(min(a[0],b[0]),min(c[0],d[0])) <= min(max(a[0],b[0]),max(c[0],d[0]))
                    and max(min(a[1],b[1]),min(c[1],d[1])) <= min(max(a[1],b[1]),max(c[1],d[1]))):
                    raise ValueError("Polygon edges cross. Draw corners in perimeter order.")


def denormalize_polygon(
    points: list[list[float]], width: int, height: int
) -> list[tuple[int, int]]:
    if len(points) < 3:
        raise ValueError("A polygon requires at least three points")
    validate_polygon(points)
    result: list[tuple[int, int]] = []
    for point in points:
        if len(point) != 2 or not all(0.0 <= value <= 1.0 for value in point):
            raise ValueError("Normalized polygon points must contain x/y values from 0 to 1")
        result.append((round(point[0] * width), round(point[1] * height)))
    return result


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    """Return True when point is inside or on the boundary of polygon."""
    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]

        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True

        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x <= intersection_x:
                inside = not inside
    return inside
