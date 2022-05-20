


def change_cion():
    from pip._vendor.distlib.compat import raw_input
    n = int( raw_input('input a number:'))
    sum = 0;
    threee_max = n//3 +1
    for i in range(threee_max):
        sum += (n - i * 3) // 2 + 1;
    print(sum)


if __name__ == '__main__':
    change_cion()