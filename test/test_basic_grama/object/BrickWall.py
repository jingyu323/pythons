from test_basic_grama.object.Wall import Wall


class BrickWall(Wall):
    def __init__(self, x, y):

        super().__init__( x, y)
        self.wall_type="brick"
        print("BrickWall init ")
