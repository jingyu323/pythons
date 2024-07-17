from test_basic_grama.object.Wall import Wall


class StoneWall(Wall):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.wall_type = "stone"
        print("StoneWall init ")
