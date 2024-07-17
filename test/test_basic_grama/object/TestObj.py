import pandas as pd
from flask import json

from test_basic_grama.object.BrickWall import BrickWall
from test_basic_grama.object.StoneWall import StoneWall


def test_pa():
     brick = BrickWall(1,2)
     stone = StoneWall(3,4)


     print( brick.wall_type,brick.x)
     print( stone.wall_type,stone.x)

     json_str = json.dumps(stone.__dict__)
     print(json_str)
     print(json.dumps(brick.__dict__))


if __name__ == '__main__':
    test_pa()




