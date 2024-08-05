class Solution:
    def networkDelayTime(self, times: list[list[int]], n: int, k: int) -> int:
        a = [[float('inf')] * n for _ in range(n)]
        for x, y, time in times:
            a[x - 1][y - 1] = time

        b = [float('inf')] * n
        b[k - 1] = 0

        c = [False] * n
        for _ in range(n):
            aa = -1
            for m, j in enumerate(c):
                if not j and (aa == -1 or (b[m] < b[aa])):
                    aa = m
            c[aa] = True
            for x, y in enumerate(a[aa]):
                b[x] = min(b[x], b[aa] + y)
        fin = max(b)
        return fin if fin < float('inf') else -1



c = Solution()
print(c.networkDelayTime([[2, 1, 1], [2, 3, 1], [3, 4, 1]], n=4, k=2))
