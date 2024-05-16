class Solution:
    def __init__(self) -> None:
        pass

    def solution(self, n, pages):
        result = None

        # TODO: 请在此编写代码
        while n > 0:
            for i in range(len(pages)):
                result = i + 1
                n = n - pages[i]
                if n <= 0:
                    break

        return result


if __name__ == "__main__":
    n = int(input().strip())

    pages = [int(item) for item in input().strip().split()]

    sol = Solution()
    result = sol.solution(n, pages)
    print(result)