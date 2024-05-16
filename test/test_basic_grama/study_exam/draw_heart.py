import turtle

# Define a function to draw a heart
def draw_heart(x, y, size):
    turtle.up()
    turtle.goto(x, y)
    turtle.down()
    turtle.begin_fill()
    turtle.color('red')
    turtle.pensize(2)
    turtle.pencolor('black')
    turtle.right(45)
    turtle.forward(size)
    turtle.circle(size/2, 180)
    turtle.left(90)
    turtle.circle(size/2, 180)
    turtle.forward(size)
    turtle.end_fill()

# Set up the turtle window and draw a heart
turtle.setup(width=500, height=500)
turtle.speed(0)
draw_heart(0, 0, 200)

# Keep the turtle window open until manually closed
turtle.done()