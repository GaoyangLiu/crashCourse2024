"""
Assignment01:网络延迟时间
采用BFS算法，寻找单源最短路径
时间复杂度：O（N²+E）
"""
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        map = [{} for i in range(n)]
        for u,v,t in times:
            map[u-1][v-1] = t
        time = [-1 for i in range(n)] 
        time[k-1] = 0

        queue = deque([[k-1,0]])
        while queue:
            cur,t = queue.popleft()
            for u ,v in map[cur].items():
                if time[u] == -1 or t+v < time[u]:
                    time[u] = t+v
                    queue.append([u,t+v])
        minT = -1
        for i in range(0,n):
            if time[i]  == -1:
                return -1
            minT = max(minT , time[i])
        return minT

        