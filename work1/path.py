import heapq
import sys


class Solution(object):
    def networkDelayTime(self, times, n, k):
        """
        :type times: List[List[int]]
        :type n: int
        :type k: int
        :rtype: int
        """
        graph = {}

        for u, v, w in times:
            graph.setdefault(u, [])
            graph[u].append([v, w])

        dist = [sys.maxsize for _ in range(n + 1)]
        dist[k] = 0

        pq = []
        heapq.heappush(pq, (dist[k], k))

        while len(pq) > 0:
            _, k = heapq.heappop(pq)

            if graph.get(k):
                for v, w in graph[k]:
                    newDist = dist[k] + w

                    if dist[v] > newDist:
                        dist[v] = newDist
                        heapq.heappush(pq, (dist[v], v))

        ans = max(dist[1:])

        return -1 if ans == sys.maxsize else ans
