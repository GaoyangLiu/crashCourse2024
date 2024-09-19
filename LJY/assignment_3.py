def sortArray(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[len(nums) // 2]
    left = [x for x in nums if x < pivot]
    middle = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]
    return sortArray(left) + middle + sortArray(right)

def main():
    arr_input = input("输入：nums = ")
    nums = [int(x) for x in arr_input.strip('[]').split(',')]
    result = sortArray(nums)
    print(result)

    input("按任意键退出...")


if __name__ == "__main__":
    main()
