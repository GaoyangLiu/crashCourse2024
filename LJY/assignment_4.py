def networkDelayTime(times,n,k):
    graph = {}
    for u, v, w in times:
        if u not in graph:
            graph[u] = []
        graph[u].append((v , w))

    Time_min = {i: float('inf') for i in range(1, n + 1)}
    Time_min[k] = 0
    Time_NtoN = [(0 , k)]

    while Time_NtoN:
        time , node = min(Time_NtoN , key = lambda x : x[0])
        Time_NtoN.remove((time , node))

        if node in graph:
            for v , w in graph[node]:
                NewTime = time + w
                if NewTime < Time_min[v]:
                    Time_min[v] = NewTime
                    Time_NtoN.append((NewTime , v))


    result = max(Time_min.values())
    return result if result < float('inf') else -1

def main():
    times_input= input("输入：times=")
    times = eval(times_input)
    n = int(input("输入 n = "))
    k = int(input("输入 k = "))

    result = networkDelayTime(times,n,k)
    print(f"输出：{result}")
    input("按任意键退出...")

if __name__ == "__main__":
    main()



